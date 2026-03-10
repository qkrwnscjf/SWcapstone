import React, { useEffect, useRef } from 'react';

const HeatmapOverlay = ({ originalImage, heatmapData, width = 224, height = 224 }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!heatmapData || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(width, height);

    // 3차원 배열 [y][x][rgba]를 Canvas용 1차원 배열로 평탄화(Flatten)
    let pos = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const [r, g, b, a] = heatmapData[y][x];
        imageData.data[pos++] = r;
        imageData.data[pos++] = g;
        imageData.data[pos++] = b;
        imageData.data[pos++] = a; // 로그에 찍힌 153(약 60% 투명도)이 그대로 적용됨
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [heatmapData, width, height]);

  return (
    <div style={{ position: 'relative', width: 'fit-content' }}>
      {/* 1. 원본 이미지 */}
      <img 
        src={originalImage} 
        alt="Original" 
        style={{ width: '400px', height: '400px', display: 'block' }} 
      />
      
      {/* 2. 히트맵 오버레이 (원본 이미지 위에 절대 위치로 배치) */}
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none', // 클릭 이벤트가 아래 이미지로 전달되게 함
          imageRendering: 'pixelated' // 히트맵이 흐릿하지 않게 처리
        }}
      />
    </div>
  );
};

export default HeatmapOverlay;