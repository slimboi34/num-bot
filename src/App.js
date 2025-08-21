import React, { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

export default function App() {
  const globalStyles = `
    *{box-sizing:border-box;margin:0;padding:0}
    body,#root{height:100%}
    .app{display:flex;align-items:center;justify-content:center;
         min-height:100vh;background:#eef2f5;padding:1rem;
         font-family:'Segoe UI',sans-serif}
    .card{position:relative;background:#fff;padding:1.5rem;border-radius:12px;
         box-shadow:0 8px 24px rgba(0,0,0,0.1);
         max-width:360px;width:100%;text-align:center}
    h1{font-size:1.8rem;margin-bottom:.4rem;color:#333}
    .sub{font-size:.9rem;color:#666;margin-bottom:1rem}
    .canvas-container{position:relative;width:100%;padding-bottom:100%;margin-bottom:1rem}
    .canvas{position:absolute;top:0;left:0;width:100%;height:100%;
           border:3px dashed #888;border-radius:8px;background:#fff;
           touch-action:none}
    .btn-group{display:flex;gap:.5rem;margin-bottom:1rem}
    .btn{flex:1;padding:.8rem;font-size:1rem;border:none;
         border-radius:6px;cursor:pointer;transition:background .2s}
    .btn-clear{background:#e74c3c;color:#fff}
    .btn-clear:hover{background:#c0392b}
    .btn-predict{background:#27ae60;color:#fff}
    .btn-predict:disabled{background:#95a5a6;cursor:not-allowed}
    .btn-predict:not(:disabled):hover{background:#1e8449}
    .btn-submit{background:#3498db;color:#fff;margin-top:.5rem}
    .btn-submit:disabled{background:#95a5a6}
    .btn-submit:hover:not(:disabled){background:#21618c}
    .result{font-size:1.2rem;color:#333;margin-bottom:1rem}
    .result .digit{font-weight:bold;font-size:1.6rem}
    .correction{display:flex;gap:.5rem;justify-content:center;margin-top:.5rem;}
    .correction input{
      width:60px;padding:.5rem;font-size:1rem;text-align:center;
      border:1px solid #ccc;border-radius:4px;
    }
    .toast{
      position:absolute; bottom:1rem; left:50%; transform:translateX(-50%);
      background:rgba(0,0,0,0.8); color:#fff; padding:.75rem 1.25rem;
      border-radius:6px; font-size:.9rem; opacity:0; animation:fadein 0.3s forwards, fadeout 0.3s forwards 2.7s;
    }
    @keyframes fadein{to{opacity:1}}
    @keyframes fadeout{to{opacity:0}}
    .loader-container{text-align:center}
    .loader{margin:2rem auto;border:6px solid #f3f3f3;
            border-top:6px solid #3498db;border-radius:50%;
            width:50px;height:50px;animation:spin 1s linear infinite}
    @keyframes spin{to{transform:rotate(360deg)}}
  `;

  const canvasRef = useRef(null);
  const isDrawing = useRef(false);
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);


  useEffect(() => {
    tf.loadGraphModel(process.env.PUBLIC_URL + '/tfjs_model/model.json')
      .then(m => setModel(m))
      .catch(err => { console.error(err); alert('Model load failed'); })
      .finally(() => {
        setLoading(false);
        clearCanvas();
      });
  }, []);

  const clearCanvas = () => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, c.width, c.height);
    setPrediction(null);
    setGrayBuffer(null);
    setCorrection('');
  };

  const getXY = e => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
    const y = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;
    return { x, y };
  };

  const startDrawing = e => {
    if (!model) return;
    e.preventDefault();
    isDrawing.current = true;
    const { x, y } = getXY(e);
    const ctx = canvasRef.current.getContext('2d');
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const draw = e => {
    if (!isDrawing.current) return;
    e.preventDefault();
    const { x, y } = getXY(e);
    const ctx = canvasRef.current.getContext('2d');
    ctx.lineWidth = 16;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#222';
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const endDrawing = () => {
    if (!isDrawing.current) return;
    isDrawing.current = false;
    canvasRef.current.getContext('2d').beginPath();
  };

  const preprocessAndPredict = async () => {
    const src = canvasRef.current;
    const wr = src.width, hr = src.height;
    const ctx = src.getContext('2d');
    const img = ctx.getImageData(0, 0, wr, hr).data;

    let xMin = wr, xMax = 0, yMin = hr, yMax = 0;
    for (let y = 0; y < hr; y++) {
      for (let x = 0; x < wr; x++) {
        if (img[(y * wr + x) * 4] < 250) {
          xMin = Math.min(xMin, x);
          xMax = Math.max(xMax, x);
          yMin = Math.min(yMin, y);
          yMax = Math.max(yMax, y);
        }
      }
    }
    if (xMax < xMin || yMax < yMin) {
      alert('Draw first!');
      return;
    }

    const boxW = xMax - xMin + 1, boxH = yMax - yMin + 1;
    const dim = Math.max(boxW, boxH);
    const off = document.createElement('canvas');
    off.width = off.height = 28;
    const octx = off.getContext('2d');
    octx.fillStyle = '#fff';
    octx.fillRect(0, 0, 28, 28);
    const pad = 2, scale = (28 - 2 * pad) / dim;
    const dx = pad + ((dim - boxW) * scale) / 2;
    const dy = pad + ((dim - boxH) * scale) / 2;
    octx.filter = 'blur(1px) contrast(200%)';
    octx.drawImage(src, xMin, yMin, boxW, boxH, dx, dy, boxW * scale, boxH * scale);
    octx.filter = 'none';

    const data = octx.getImageData(0, 0, 28, 28).data;
    const gray = new Float32Array(784);
    for (let i = 0; i < 784; i++) {
      const v = 255 - data[i * 4];
      gray[i] = v > 50 ? 1 : 0;
    }
    setGrayBuffer(Array.from(gray));

    const input = tf.tensor(gray, [1, 28, 28, 1]);
    const p = model.predict(input).argMax(-1).dataSync()[0];
    setPrediction(p);
  };


  if (loading) {
    return (
      <div className="loader-container">
        <style>{globalStyles}</style>
        <div className="loader" />
        <p>Loading…</p>
      </div>
    );
  }

  return (
    <div className="app">
      <style>{globalStyles}</style>
      <div className="card">
        <h1>🧠 Digit Recogniser</h1>
        <p className="sub">Draw a digit (0–9), then Predict</p>
        <div className="canvas-container">
          <canvas
            ref={canvasRef} width={280} height={280}
            className="canvas"
            onMouseDown={startDrawing} onMouseMove={draw}
            onMouseUp={endDrawing} onMouseLeave={endDrawing}
            onTouchStart={startDrawing}
            onTouchMove={draw}
            onTouchEnd={endDrawing}
          />
        </div>
        <div className="btn-group">
          <button onClick={clearCanvas} className="btn btn-clear">Clear</button>
          <button onClick={preprocessAndPredict} disabled={!model} className="btn btn-predict">Predict</button>
        </div>
        {prediction !== null && (
          <>
            <div className="result">
              Prediction: <span className="digit">{prediction}</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}