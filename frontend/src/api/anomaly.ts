import type { PredictResponse } from "../types/anomaly";

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export async function predictAnomaly(file: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("file", file);

  let res: Response;
  try {
    res = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
    });
  } catch (error) {
    if (error instanceof TypeError) {
      throw new Error("백엔드 연결에 실패했습니다. API 서버가 실행 중인지 확인해 주세요.");
    }
    throw error;
  }

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }

  return res.json();
}
