import React, { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import './App.css';

function App() {
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);

  // Load model on mount
  useEffect(() => {
    async function loadModel() {
      try {
        // Load the model from public/tfjs_model/model.json
        const loadedModel = await tf.loadGraphModel(process.env.PUBLIC_URL + '/tfjs_model/model.json');
        console.log("Model loaded successfully!");
        setModel(loadedModel);
      } catch (err) {
        console.error("Failed to load model", err);
      }
    }
    loadModel();
  }, []);

  // Clear the canvas
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'white';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
    setPrediction(null);
  };

  // Drawing handlers
  let isDrawing = false;
  const startDrawing = (e) => {
    isDrawing = true;
    draw(e);
  };
  
  const endDrawing = () => {
    isDrawing = false;
    const canvas = canvasRef.current;
    if (canvas) {
      canvas.getContext('2d').beginPath();
    }
  };
  
  const draw = (e) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    // Calculate mouse position relative to the canvas
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
  
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';
  
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  // Preprocess the canvas image and predict digit
  const predictDigit = async () => {
    if (!model) {
      alert("Model not loaded yet!");
      return;
    }
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    // Create an offscreen canvas to resize the drawing to 28x28
    const offscreen = document.createElement('canvas');
    offscreen.width = 28;
    offscreen.height = 28;
    const offCtx = offscreen.getContext('2d');

    // Draw the main canvas content onto the offscreen canvas.
    offCtx.drawImage(canvas, 0, 0, 28, 28);
    let imageData = offCtx.getImageData(0, 0, 28, 28);
    
    // Convert to grayscale and normalize.
    // Assuming the drawing uses black on a white background:
    const data = imageData.data;
    const grayData = [];
    for (let i = 0; i < data.length; i += 4) {
      // Use red channel (all channels are equal in grayscale), then invert
      const inverted = 255 - data[i]; // invert so the drawn digit becomes white (high value)
      grayData.push(inverted / 255);
    }
    
    // Create a tensor of shape [1, 28, 28, 1]
    const input = tf.tensor(grayData, [1, 28, 28, 1]);
    // Run prediction
    const predictionTensor = model.predict(input);
    const predictedDigit = predictionTensor.argMax(-1).dataSync()[0];
    setPrediction(predictedDigit);
  };

  return (
    <div className="App">
      <h1>Digit Recognizer</h1>
      <canvas
        ref={canvasRef}
        width={280}
        height={280}
        style={{ border: '1px solid #000', background: 'white' }}
        onMouseDown={startDrawing}
        onMouseMove={draw}
        onMouseUp={endDrawing}
        onMouseLeave={endDrawing}
      />
      <div style={{ marginTop: '20px' }}>
        <button onClick={clearCanvas}>Clear</button>
        <button onClick={predictDigit} style={{ marginLeft: '10px' }}>Predict</button>
      </div>
      {prediction !== null && <h2>Prediction: {prediction}</h2>}
    </div>
  );
}

export default App;