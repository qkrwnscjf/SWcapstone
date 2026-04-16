"""
PatchCore-based anomaly detection model for heatmap generation.

Implements the PatchCore algorithm (Roth et al., 2022) for unsupervised
anomaly detection using pre-trained CNN backbones. The model extracts
patch-level features from intermediate network layers, builds a coreset
memory bank from normal training images, and uses k-NN distance to
produce spatial anomaly heatmaps at inference time.

Usage:
    from src.heatmap_model import PatchCoreModel

    model = PatchCoreModel(backbone_name="wide_resnet50_2", device="mps")
    model.fit(train_dataloader)
    result = model.predict(image_tensor)
    # result["anomaly_score"]            -> float
    # result["raw_score_heatmap"]        -> np.ndarray (H, W)
    # result["normalized_score_heatmap"] -> np.ndarray (H, W) in [0, 1]
    # result["heatmap_overlay"]          -> np.ndarray (H, W, 4) RGBA
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from torchvision import models

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INPUT_SIZE: int = 224
IMAGENET_MEAN: Tuple[float, ...] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, ...] = (0.229, 0.224, 0.225)

# Backbone registry: maps name -> (constructor, weight enum, layer2 name, layer3 name)
_BACKBONE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "resnet18": {
        "factory": models.resnet18,
        "weights": models.ResNet18_Weights.DEFAULT,
        "layer_names": ("layer2", "layer3"),
    },
    "wide_resnet50_2": {
        "factory": models.wide_resnet50_2,
        "weights": models.Wide_ResNet50_2_Weights.DEFAULT,
        "layer_names": ("layer2", "layer3"),
    },
}


# ---------------------------------------------------------------------------
# Helper: feature extraction hooks
# ---------------------------------------------------------------------------
class _FeatureExtractor(torch.nn.Module):
    """Wraps a torchvision backbone and captures intermediate feature maps.

    Registers forward hooks on the specified layers to collect activations
    during each forward pass.
    """

    def __init__(self, backbone: torch.nn.Module, layer_names: Tuple[str, ...]) -> None:
        super().__init__()
        self.backbone = backbone
        self.layer_names = layer_names
        self._features: Dict[str, torch.Tensor] = {}
        self._hooks: list = []

        for name in layer_names:
            layer = dict(self.backbone.named_children())[name]
            hook = layer.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

    def _make_hook(self, name: str):
        def hook_fn(_module, _input, output):
            self._features[name] = output
        return hook_fn

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run a forward pass and return captured feature maps.

        Args:
            x: Input tensor of shape (B, 3, 224, 224).

        Returns:
            Dictionary mapping layer names to their feature tensors.
        """
        self._features.clear()
        self.backbone(x)
        return dict(self._features)

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ---------------------------------------------------------------------------
# Coreset subsampling (greedy farthest-point)
# ---------------------------------------------------------------------------
def _coreset_subsample(
    features: np.ndarray,
    ratio: float,
    seed: int = 42,
) -> np.ndarray:
    """Select a coreset via greedy farthest-point sampling.

    This provides a representative subset of the full feature memory bank
    while significantly reducing storage and inference cost.

    Args:
        features: Array of shape (N, D) containing all patch features.
        ratio: Fraction of features to keep, in (0, 1].
        seed: Random seed for the initial point selection.

    Returns:
        Subsampled feature array of shape (M, D) where M = ceil(N * ratio).
    """
    n = features.shape[0]
    m = max(1, int(np.ceil(n * ratio)))
    if m >= n:
        return features.copy()

    rng = np.random.RandomState(seed)
    selected_indices: List[int] = [rng.randint(0, n)]
    min_distances = np.full(n, np.inf, dtype=np.float64)

    for _ in range(m - 1):
        last = features[selected_indices[-1]]
        dists = np.linalg.norm(features - last[np.newaxis, :], axis=1)
        min_distances = np.minimum(min_distances, dists)
        next_idx = int(np.argmax(min_distances))
        selected_indices.append(next_idx)

    return features[selected_indices]


# ---------------------------------------------------------------------------
# PatchCoreModel
# ---------------------------------------------------------------------------
class PatchCoreModel:
    """PatchCore anomaly detection model.

    Extracts multi-scale patch embeddings from a pre-trained CNN backbone,
    builds a coreset memory bank from normal images, and scores test images
    via k-nearest-neighbour distance in the embedding space.

    Attributes:
        backbone_name: Name of the backbone architecture.
        device: Torch device string ("cpu", "cuda", "mps").
        coreset_ratio: Fraction of patch features retained during coreset
            subsampling.
        k_neighbors: Number of nearest neighbours used for scoring.
        input_size: Expected spatial input size (square).
    """

    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        device: Optional[str] = None,
        coreset_ratio: float = 0.10,
        k_neighbors: int = 9,
        input_size: int = INPUT_SIZE,
    ) -> None:
        """Initialise the PatchCore model.

        Args:
            backbone_name: One of "resnet18" or "wide_resnet50_2".
            device: Target device. If ``None``, automatically selects MPS,
                CUDA, or CPU in that order.
            coreset_ratio: Ratio of patch features to keep via farthest-point
                coreset sampling. Must be in (0, 1].
            k_neighbors: Number of nearest neighbours for anomaly scoring.
            input_size: Spatial size of input images (assumed square).

        Raises:
            ValueError: If ``backbone_name`` is not supported or
                ``coreset_ratio`` is out of range.
        """
        if backbone_name not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                f"Choose from {list(_BACKBONE_REGISTRY.keys())}."
            )
        if not 0.0 < coreset_ratio <= 1.0:
            raise ValueError(f"coreset_ratio must be in (0, 1], got {coreset_ratio}.")

        self.backbone_name = backbone_name
        self.coreset_ratio = coreset_ratio
        self.k_neighbors = k_neighbors
        self.input_size = input_size

        # Resolve device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device

        # Build backbone
        cfg = _BACKBONE_REGISTRY[backbone_name]
        backbone = cfg["factory"](weights=cfg["weights"])
        backbone.eval()
        self._feature_extractor = _FeatureExtractor(backbone, cfg["layer_names"])
        self._feature_extractor.to(self.device)
        self._layer_names: Tuple[str, ...] = cfg["layer_names"]

        # Memory bank (populated by .fit())
        self._memory_bank: Optional[np.ndarray] = None  # (M, D)
        self._nn_index: Optional[NearestNeighbors] = None

        # Spatial dimensions of the feature map at the finest resolution
        # (set during fit based on actual forward pass)
        self._feature_map_h: int = 0
        self._feature_map_w: int = 0
        self._embedding_dim: int = 0

        logger.info(
            "PatchCoreModel initialised: backbone=%s, device=%s, "
            "coreset_ratio=%.2f, k=%d",
            backbone_name, self.device, coreset_ratio, k_neighbors,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract and concatenate multi-scale patch features.

        Takes feature maps from layer2 and layer3, up-samples layer3 to
        match layer2's spatial resolution, and concatenates along the
        channel dimension. The result is reshaped to (B * H * W, D).

        Args:
            x: Batch of images, shape (B, 3, H, W).

        Returns:
            Patch features of shape (B * H * W, D).
        """
        feat_dict = self._feature_extractor(x)
        layer2_name, layer3_name = self._layer_names
        feat2 = feat_dict[layer2_name]  # (B, C2, H2, W2)
        feat3 = feat_dict[layer3_name]  # (B, C3, H3, W3)

        # Up-sample layer3 features to layer2 spatial size
        feat3_up = F.interpolate(
            feat3, size=feat2.shape[2:], mode="bilinear", align_corners=False
        )

        # Concatenate along channel axis
        combined = torch.cat([feat2, feat3_up], dim=1)  # (B, C2+C3, H2, W2)

        B, D, H, W = combined.shape
        # Reshape to patch-level: (B*H*W, D)
        patches = combined.permute(0, 2, 3, 1).reshape(-1, D)
        return patches, H, W, D

    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Move a tensor to CPU and convert to numpy, handling MPS."""
        if tensor.device.type == "mps":
            return tensor.cpu().numpy()
        return tensor.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(self, dataloader: DataLoader) -> "PatchCoreModel":
        """Train the model by extracting a memory bank from normal images.

        Iterates over all batches in the dataloader, extracts patch-level
        features, then reduces them via coreset subsampling and builds
        a k-NN index for fast scoring.

        The dataloader should yield tensors (or tuples whose first element
        is a tensor) of shape (B, 3, 224, 224) containing *normal* images.

        Args:
            dataloader: PyTorch DataLoader supplying normal training images.

        Returns:
            self, for method chaining.
        """
        self._feature_extractor.eval()
        all_patches: List[np.ndarray] = []

        logger.info("Extracting features from training set (%d batches)...", len(dataloader))

        for batch_idx, batch in enumerate(dataloader):
            # Support both plain tensors and (image, label) tuples
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(self.device)

            patches, H, W, D = self._extract_features(images)
            all_patches.append(self._to_numpy(patches))

            if (batch_idx + 1) % 10 == 0:
                logger.info("  Processed batch %d / %d", batch_idx + 1, len(dataloader))

        # Store spatial dimensions from the last batch
        self._feature_map_h = H
        self._feature_map_w = W
        self._embedding_dim = D

        all_patches_np = np.concatenate(all_patches, axis=0)  # (N_total, D)
        logger.info(
            "Total patch features: %d (dim=%d). Running coreset subsampling (ratio=%.2f)...",
            all_patches_np.shape[0], all_patches_np.shape[1], self.coreset_ratio,
        )

        # Coreset subsampling
        self._memory_bank = _coreset_subsample(
            all_patches_np, self.coreset_ratio, seed=42,
        )
        logger.info("Coreset size: %d", self._memory_bank.shape[0])

        # Build k-NN index (always on CPU / numpy)
        self._nn_index = NearestNeighbors(
            n_neighbors=self.k_neighbors,
            metric="euclidean",
            algorithm="auto",
        )
        self._nn_index.fit(self._memory_bank)
        logger.info("k-NN index built (k=%d). Training complete.", self.k_neighbors)

        return self

    # ------------------------------------------------------------------
    # predict (single image)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Compute anomaly score and heatmaps for a single image.

        Args:
            image_tensor: A tensor of shape (3, 224, 224) or (1, 3, 224, 224).
                Should be normalised with ImageNet statistics.

        Returns:
            Dictionary with the following keys:

            - ``anomaly_score`` (float): Image-level anomaly score (max of
              the patch-level k-NN distances).
            - ``raw_score_heatmap`` (np.ndarray): Spatial anomaly map of
              shape (224, 224) with raw k-NN distance values.
            - ``normalized_score_heatmap`` (np.ndarray): Min-max normalised
              version of the raw heatmap, values in [0, 1].
            - ``heatmap_overlay`` (np.ndarray): RGBA image (224, 224, 4) with
              a jet-colourmap overlay suitable for visualisation.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self._memory_bank is None or self._nn_index is None:
            raise RuntimeError(
                "Model has not been fitted. Call .fit(dataloader) first."
            )

        # Ensure batch dimension
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        patches, H, W, D = self._extract_features(image_tensor)
        patches_np = self._to_numpy(patches)  # (H*W, D)

        # k-NN distances (on CPU)
        distances, _ = self._nn_index.kneighbors(patches_np)  # (H*W, k)
        # Use the mean of k-NN distances as the patch score
        patch_scores = distances.mean(axis=1)  # (H*W,)

        # Reshape to spatial map
        score_map = patch_scores.reshape(H, W)  # (H_feat, W_feat)

        # Up-sample to input size using bilinear interpolation
        raw_heatmap = self._upsample_score_map(score_map)  # (224, 224)

        # Smooth with Gaussian to reduce patchiness
        raw_heatmap = gaussian_filter(raw_heatmap, sigma=4.0)

        # Image-level score: maximum patch anomaly score
        anomaly_score = float(raw_heatmap.max())

        # Normalised heatmap [0, 1]
        normalized_heatmap = self._normalize_map(raw_heatmap)

        # Overlay (RGBA jet colourmap)
        heatmap_overlay = self._make_overlay(normalized_heatmap)

        return {
            "anomaly_score": anomaly_score,
            "raw_score_heatmap": raw_heatmap,
            "normalized_score_heatmap": normalized_heatmap,
            "heatmap_overlay": heatmap_overlay,
        }

    # ------------------------------------------------------------------
    # predict_batch
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_batch(self, dataloader: DataLoader) -> List[Dict[str, Any]]:
        """Run prediction on an entire dataloader.

        Processes images batch-by-batch for efficiency, but returns per-image
        results identical to calling :meth:`predict` individually.

        Args:
            dataloader: DataLoader yielding image tensors (or tuples whose
                first element is the image tensor).

        Returns:
            List of result dictionaries, one per image, with the same keys
            as :meth:`predict`.
        """
        if self._memory_bank is None or self._nn_index is None:
            raise RuntimeError(
                "Model has not been fitted. Call .fit(dataloader) first."
            )

        results: List[Dict[str, Any]] = []

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images = batch[0]
            else:
                images = batch
            images = images.to(self.device)

            patches, H, W, D = self._extract_features(images)
            patches_np = self._to_numpy(patches)  # (B*H*W, D)

            distances, _ = self._nn_index.kneighbors(patches_np)
            patch_scores = distances.mean(axis=1)

            B = images.shape[0]
            patch_scores = patch_scores.reshape(B, H, W)

            for i in range(B):
                raw_heatmap = self._upsample_score_map(patch_scores[i])
                raw_heatmap = gaussian_filter(raw_heatmap, sigma=4.0)
                anomaly_score = float(raw_heatmap.max())
                normalized_heatmap = self._normalize_map(raw_heatmap)
                heatmap_overlay = self._make_overlay(normalized_heatmap)

                results.append({
                    "anomaly_score": anomaly_score,
                    "raw_score_heatmap": raw_heatmap,
                    "normalized_score_heatmap": normalized_heatmap,
                    "heatmap_overlay": heatmap_overlay,
                })

        return results

    # ------------------------------------------------------------------
    # Heatmap utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _upsample_score_map(score_map: np.ndarray) -> np.ndarray:
        """Bilinear up-sample a (H_feat, W_feat) score map to (224, 224).

        Args:
            score_map: 2-D array of patch-level scores.

        Returns:
            Up-sampled 2-D array of shape (224, 224).
        """
        t = torch.from_numpy(score_map).unsqueeze(0).unsqueeze(0).float()
        t_up = F.interpolate(
            t, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False,
        )
        return t_up.squeeze().numpy()

    @staticmethod
    def _normalize_map(score_map: np.ndarray) -> np.ndarray:
        """Min-max normalise a score map to [0, 1].

        Args:
            score_map: 2-D array of raw anomaly scores.

        Returns:
            Normalised 2-D array with values in [0, 1].
        """
        smin = score_map.min()
        smax = score_map.max()
        if smax - smin < 1e-8:
            return np.zeros_like(score_map, dtype=np.float32)
        return ((score_map - smin) / (smax - smin)).astype(np.float32)

    @staticmethod
    def _make_overlay(
        normalized_map: np.ndarray, alpha: float = 0.6
    ) -> np.ndarray:
        """Create an RGBA overlay image from a normalised heatmap.

        Uses a jet-style colourmap (computed without matplotlib dependency)
        so that low scores are blue and high scores are red.

        Args:
            normalized_map: 2-D array with values in [0, 1].
            alpha: Opacity of the overlay, in [0, 1].

        Returns:
            RGBA image of shape (H, W, 4) with dtype uint8.
        """
        # Jet colourmap approximation via linear interpolation
        v = normalized_map.clip(0, 1)
        r = np.clip(1.5 - np.abs(v - 0.75) * 4.0, 0, 1)
        g = np.clip(1.5 - np.abs(v - 0.50) * 4.0, 0, 1)
        b = np.clip(1.5 - np.abs(v - 0.25) * 4.0, 0, 1)
        a = np.full_like(v, alpha)

        overlay = np.stack([r, g, b, a], axis=-1)
        return (overlay * 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        """Save the model state (coreset memory bank and configuration).

        The backbone weights are not saved because they come from
        torchvision pretrained checkpoints and can be reconstructed.

        Args:
            path: File path to write the pickle state to.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            "backbone_name": self.backbone_name,
            "coreset_ratio": self.coreset_ratio,
            "k_neighbors": self.k_neighbors,
            "input_size": self.input_size,
            "memory_bank": self._memory_bank,
            "feature_map_h": self._feature_map_h,
            "feature_map_w": self._feature_map_w,
            "embedding_dim": self._embedding_dim,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path], device: Optional[str] = None) -> "PatchCoreModel":
        """Load a previously saved PatchCore model.

        Reconstructs the backbone from torchvision, restores the coreset
        memory bank, and rebuilds the k-NN index.

        Args:
            path: Path to the saved pickle file.
            device: Target device. If ``None``, auto-detected.

        Returns:
            A ready-to-use PatchCoreModel instance.

        Raises:
            FileNotFoundError: If the path does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(path, "rb") as f:
            # 수정 전
            # state = pickle.load(f)
            # 수정 후 (더 안전한 로딩 방식 시도)
            # 수정 후: state 변수를 미리 선언하거나 try 블록 전체를 아래처럼 감싸기
        
            state = None # 변수를 미리 선언
            try:
                state = torch.load(path, map_location=device) 
                logger.info("Heatmap model loaded using torch.load")
            except Exception as e:
                # torch.load가 실패할 경우의 대비책
                logger.warning(f"torch.load failed, trying pickle: {e}")
                with open(path, "rb") as f:
                    state = pickle.load(f)

        """"
        model = cls(
            backbone_name=state["backbone_name"],
            device=device,
            coreset_ratio=state["coreset_ratio"],
            #k_neighbors=state["k_neighbors"],
            k_neighbors=state.get("k_neighbors", 9),
            input_size=state["input_size"],
        )
        """
        backbone_name = state.get("backbone_name", "resnet18")
        input_size = state.get("input_size", (224, 224))
        k_neighbors = state.get("k_neighbors", 9)

        # 2. 클래스 생성자(__init__)를 호출
        # 'layers'가 문제였 - 일단 제외하고 생성
        # 1. 모델 객체 생성 (기존과 동일하거나 .get 적용)
        model = cls(
            backbone_name=state.get("backbone_name", "resnet18"),
            input_size=state.get("input_size", (224, 224)),
            k_neighbors=state.get("k_neighbors", 9),
            device=device
        )

        # 2. 내부 변수 할당 (KeyError 방지 로직)
        # memory_bank가 coreset이라는 이름으로 저장되어 있을 수도 있으니 둘 다 체크
        model._memory_bank = state.get("memory_bank", state.get("coreset", None))
        
        # 특징 맵 크기 및 임베딩 차원 (PatchCore ResNet18 기본값: 28, 28, 448)
        model._feature_map_h = state.get("feature_map_h", 28)
        model._feature_map_w = state.get("feature_map_w", 28)
        model._embedding_dim = state.get("embedding_dim", 448)

        # 3. k-NN 인덱스 재구축
        if model._memory_bank is not None:
            from sklearn.neighbors import NearestNeighbors
            model._nn_index = NearestNeighbors(
                n_neighbors=model.k_neighbors,
                metric="euclidean",
                algorithm="auto",
            )
            model._nn_index.fit(model._memory_bank)
            logger.info(
                "Model loaded from %s (coreset=%d, k=%d)",
                path, model._memory_bank.shape[0], model.k_neighbors,
            )
        else:
            logger.warning("Memory bank is None. Heatmap generation might not work properly.")

        return model
        """
        model = cls(
            backbone_name=backbone_name,
            input_size=input_size,
            k_neighbors=k_neighbors,
            device=device
        )
        model._memory_bank = state["memory_bank"]
        model._feature_map_h = state["feature_map_h"]
        model._feature_map_w = state["feature_map_w"]
        model._embedding_dim = state["embedding_dim"]

        # Rebuild k-NN index
        if model._memory_bank is not None:
            model._nn_index = NearestNeighbors(
                n_neighbors=model.k_neighbors,
                metric="euclidean",
                algorithm="auto",
            )
            model._nn_index.fit(model._memory_bank)
            logger.info(
                "Model loaded from %s (coreset=%d, k=%d)",
                path, model._memory_bank.shape[0], model.k_neighbors,
            )

        return model
        """

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        fitted = self._memory_bank is not None
        coreset_size = self._memory_bank.shape[0] if fitted else 0
        return (
            f"PatchCoreModel("
            f"backbone={self.backbone_name!r}, "
            f"device={self.device!r}, "
            f"coreset_ratio={self.coreset_ratio}, "
            f"k={self.k_neighbors}, "
            f"fitted={fitted}, "
            f"coreset_size={coreset_size})"
        )
