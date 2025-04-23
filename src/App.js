import React, { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import './App.css';

function App() {
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    async function loadModel() {
      try {
        const loadedModel = await tf.loadGraphModel(process.env.PUBLIC_URL + '/tfjs_model/model.json');
        console.log("✅ Model loaded successfully!");
        setModel(loadedModel);
      } catch (err) {
        console.error("❌ Failed to load model", err);
      }
    }
    loadModel();
  }, []);

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    setPrediction(null);
  };

  let isDrawing = false;

  const getCoordinates = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    if (e.touches) {
      const touch = e.touches[0];
      return { x: touch.clientX - rect.left, y: touch.clientY - rect.top };
    } else {
      return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    }
  };

  const startDrawing = (e) => {
    isDrawing = true;
    const { x, y } = getCoordinates(e);
    const ctx = canvasRef.current.getContext('2d');
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const draw = (e) => {
    if (!isDrawing) return;
    e.preventDefault(); // Prevent scrolling on mobile
    const { x, y } = getCoordinates(e);
    const ctx = canvasRef.current.getContext('2d');
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const endDrawing = () => {
    isDrawing = false;
    const canvas = canvasRef.current;
    canvas.getContext('2d').beginPath();
  };

  const predictDigit = async () => {
    if (!model) {
      alert("Model not loaded yet!");
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;

    const offscreen = document.createElement('canvas');
    offscreen.width = 28;
    offscreen.height = 28;
    const offCtx = offscreen.getContext('2d');
    offCtx.drawImage(canvas, 0, 0, 28, 28);
    const imageData = offCtx.getImageData(0, 0, 28, 28);

    const data = imageData.data;
    const grayData = [];
    for (let i = 0; i < data.length; i += 4) {
      const inverted = 255 - data[i];
      grayData.push(inverted / 255);
    }

    const input = tf.tensor(grayData, [1, 28, 28, 1]);
    const predictionTensor = model.predict(input);
    const predictedDigit = predictionTensor.argMax(-1).dataSync()[0];

    setPrediction(predictedDigit);
  };

  return (
    <div className="App" style={{ padding: '1rem', fontFamily: 'Arial, sans-serif', maxWidth: 600, margin: 'auto' }}>
      <h1 style={{ fontSize: '1.8rem' }}>🧠 Digit Recogniser</h1>
      <p style={{ fontSize: '1rem' }}>Draw a digit (0–9) below and tap <strong>Predict</strong>.</p>

      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        style={{
          border: '2px solid #000',
          background: 'white',
          borderRadius: '8px',
          touchAction: 'none' // Important for mobile drawing
        }}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={endDrawing}
        onMouseLeave={endDrawing}
        onTouchStart={startDrawing}
        onTouchMove={draw}
        onTouchEnd={endDrawing}
      />

      <div style={{ marginTop: '20px', display: 'flex', gap: '10px', justifyContent: 'center', flexWrap: 'wrap' }}>
        <button
          onClick={clearCanvas}
          style={{
            padding: '12px 20px',
            fontSize: '16px',
            borderRadius: '6px',
            backgroundColor: '#f44336',
            color: 'white',
            border: 'none',
            cursor: 'pointer',
            flexGrow: 1,
            minWidth: '120px'
          }}
        >
          Clear
        </button>
        <button
          onClick={predictDigit}
          disabled={!model}
          style={{
            padding: '12px 20px',
            fontSize: '16px',
            borderRadius: '6px',
            backgroundColor: model ? '#4CAF50' : '#9e9e9e',
            color: 'white',
            border: 'none',
            cursor: model ? 'pointer' : 'not-allowed',
            flexGrow: 1,
            minWidth: '120px'
          }}
        >
          Predict
        </button>
      </div>

      {prediction !== null && (
        <div style={{ marginTop: '20px', fontSize: '1.5rem', textAlign: 'center' }}>
          <strong>Prediction:</strong> {prediction}
        </div>
      )}
    </div>
  );
}

export default App;