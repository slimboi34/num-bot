import React, { useRef, useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';
import './App.css';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LabelList,
  ScatterChart, Scatter
} from 'recharts';

function App() {
  const canvasRef = useRef(null);
  const [model, setModel] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [probabilities, setProbabilities] = useState([]);
  const [accuracyData, setAccuracyData] = useState([]);

  useEffect(() => {
    async function loadModel() {
      try {
        const loadedModel = await tf.loadGraphModel(process.env.PUBLIC_URL + '/tfjs_model/model.json');
        console.log("Model loaded successfully!");
        setModel(loadedModel);
      } catch (err) {
        console.error("Failed to load model", err);
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
    setProbabilities([]);
    setAccuracyData([]);
  };

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
    let imageData = offCtx.getImageData(0, 0, 28, 28);

    const data = imageData.data;
    const grayData = [];
    for (let i = 0; i < data.length; i += 4) {
      const inverted = 255 - data[i];
      grayData.push(inverted / 255);
    }

    const input = tf.tensor(grayData, [1, 28, 28, 1]);
    const predictionTensor = model.predict(input);
    const predictionArray = await predictionTensor.data();
    const predictedDigit = predictionTensor.argMax(-1).dataSync()[0];

    setPrediction(predictedDigit);

    const formattedData = predictionArray.map((prob, index) => ({
      digit: index,
      probability: parseFloat((prob * 100).toFixed(2))
    }));

    setProbabilities(formattedData);
    setAccuracyData(formattedData); // Accuracy = same as prediction probability
  };

  return (
    <div className="App">
      <h1>Digit Recogniser</h1>
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

      {/* Bar chart */}
      {probabilities.length > 0 && (
        <div style={{ width: '100%', maxWidth: '600px', height: 300, margin: 'auto' }}>
          <h3>Prediction Probabilities</h3>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={probabilities}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="digit" />
              <YAxis unit="%" />
              <Tooltip />
              <Bar dataKey="probability" fill="#8884d8">
                <LabelList dataKey="probability" position="top" />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Scatter chart */}
      {accuracyData.length > 0 && (
        <div style={{ width: '100%', maxWidth: '600px', height: 300, margin: 'auto', marginTop: 40 }}>
          <h3>Confidence Per Digit (Scatter Plot)</h3>
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart>
              <CartesianGrid />
              <XAxis dataKey="digit" name="Digit" />
              <YAxis dataKey="probability" name="Confidence" unit="%" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Scatter name="Confidence" data={accuracyData} fill="#82ca9d" />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

export default App;