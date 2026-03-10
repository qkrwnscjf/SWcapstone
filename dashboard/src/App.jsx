import { useState, useEffect, useRef } from 'react';
import './App.css';

function App() {
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const canvasRef = useRef(null); // 히트맵을 그릴 캔버스 참조

  // 히트맵 데이터가 올 때마다 캔버스에 그리기
  useEffect(() => {
    if (result?.heatmap_data && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const data = result.heatmap_data; // [224][224][4]
      const height = data.length;
      const width = data[0].length;

      const imageData = ctx.createImageData(width, height);
      let pos = 0;
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const [r, g, b, a] = data[y][x];
          imageData.data[pos++] = r;
          imageData.data[pos++] = g;
          imageData.data[pos++] = b;
          imageData.data[pos++] = a;
        }
      }
      ctx.putImageData(imageData, 0, 0);
    }
  }, [result]);

  const onUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setResult(null); 
      setPreview(URL.createObjectURL(file));
      startInference(file);
    }
  };

  const startInference = async (file) => {
    setLoading(true);
    const form = new FormData();
    form.append('file', file);
    try {
      const res = await fetch('http://localhost:8000/predict', { method: 'POST', body: form });
      const data = await res.json();
      setResult(data);
    } catch (e) { 
      alert("서버 연결 실패"); 
    } finally { 
      setLoading(false); 
    }
  };

  return (
    <div className="hmi-container">
      <header>
        <h1 style={{fontSize:'1.1rem', color:'var(--blue)'}}>STEEL QUALITY MONITORING [ROUND 3]</h1>
        <div style={{fontSize:'0.8rem'}}>STATUS: <span style={{color: loading ? 'orange' : 'var(--normal)'}}>{loading ? 'SCANNING...' : 'READY'}</span></div>
      </header>

      <main className="monitor-panel" style={{position: 'relative'}}>
        <div className="re-scan-overlay">
          <input type="file" id="re-up" onChange={onUpload} hidden />
          <label htmlFor="re-up" className="btn-secondary">
            {preview ? "NEW SCAN" : "SELECT IMAGE"}
          </label>
        </div>

        {preview ? (
          <div className="image-wrapper" style={{position: 'relative', width: '100%', height: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
            {/* 원본 이미지 */}
            <img src={preview} alt="Inspection view" style={{maxWidth: '100%', maxHeight: '100%', objectFit: 'contain'}} />
            
            {/* 결함 위치 표시 (Canvas 기반 Heatmap) */}
            {result?.heatmap_data && (
              <canvas 
                ref={canvasRef}
                width={224}
                height={224}
                className="heatmap-layer"
                style={{
                  position: 'absolute',
                  width: '100%', // 원본 이미지 크기에 맞춤
                  height: '100%',
                  top: 0,
                  left: 0,
                  pointerEvents: 'none',
                  opacity: 0.6, // 투명도 조절
                  imageRendering: 'pixelated',
                  objectFit: 'contain'
                }}
              />
            )}
            {loading && <div className="scan-line"></div>}
          </div>
        ) : (
          <p style={{color:'#475569'}}>이미지를 업로드하여 검사를 시작하십시오.</p>
        )}
      </main>

      <aside className="side-panel">
        <div className="card">
          <h3 style={{margin:0, fontSize:'0.85rem', color:'var(--text-sub)'}}>DIAGNOSIS</h3>
          {result ? (
            <div className="status-badge" style={{
              borderColor: result.decision.includes('anomaly') ? 'var(--anomaly)' : 'var(--normal)',
              color: result.decision.includes('anomaly') ? 'var(--anomaly)' : 'var(--normal)'
            }}>
              {result.decision.toUpperCase()}
            </div>
          ) : <div className="status-badge" style={{borderColor:'#e2e8f0', color:'#cbd5e1'}}>IDLE</div>}
          
          <div style={{marginTop:'20px', fontSize:'0.8rem'}}>
             <div className="metric-row">Gate Prob: <strong>{(result?.gate_score*100 || 0).toFixed(2)}%</strong></div>
             <div className="metric-row">Heatmap: <strong>{result?.heatmap_score?.toFixed(4) || "0.0000"}</strong></div>
          </div>
        </div>
        
        <div className="card" style={{flex:0.3, fontSize:'0.7rem', color:'var(--text-sub)'}}>
          <p>Model: PatchCore (ResNet-18)</p>
          <p>Backbone: {result?.mlflow_info?.backbone || "N/A"}</p>
          <p>Threshold: {result?.mlflow_info?.threshold || "3.8288"}</p>
        </div>
      </aside>

      <footer className="bottom-panel">
        <h3 style={{margin:0, fontSize:'0.8rem', marginBottom:'10px'}}>INSPECTION_HISTORY_LOG</h3>
        <div style={{display:'flex', gap:'10px', overflowX:'auto'}}>
          {result?.mlflow_info?.history ? (
            result.mlflow_info.history.map((log, i) => (
              <div key={i} style={{padding:'8px', background:'#f1f5f9', fontSize:'0.65rem', border:'1px solid #e2e8f0', minWidth:'120px'}}>
                EP_{log.epoch} | F1: {log.val_f1?.toFixed(3)}
              </div>
            ))
          ) : <p style={{fontSize:'0.6rem'}}>No history data available.</p>}
        </div>
      </footer>
    </div>
  );
}

export default App;