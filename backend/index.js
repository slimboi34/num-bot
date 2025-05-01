const express = require('express');
const cors    = require('cors');
const fs      = require('fs');
const path    = require('path');

const app = express();
app.use(cors());
app.use(express.json());

const DATA_DIR = path.join(__dirname, 'data');
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR);

app.post('/api/save-data', (req, res) => {
  const { grayData, label } = req.body;
  if (!Array.isArray(grayData) || typeof label !== 'number') {
    return res.status(400).json({ error: 'Invalid payload' });
  }
  const fname = `${Date.now()}_${label}.json`;
  fs.writeFileSync(
    path.join(DATA_DIR, fname),
    JSON.stringify({ grayData, label })
  );
  res.json({ status: 'saved' });
});

const PORT = process.env.PORT || 3001;
app.listen(PORT, () => {
  console.log(`✅ Backend listening on http://localhost:${PORT}`);
});