import express, { Request, Response, NextFunction } from 'express';
import path from 'path';
import dotenv from 'dotenv';
import cors from 'cors';
import fetch from 'node-fetch'; // Import fetch for making HTTP requests

// Load environment variables
dotenv.config();

// Initialize Express application
const app = express();
const PORT = process.env.PORT || 3000;

// Scene Analysis Service URL
const SCENE_ANALYZER_URL = process.env.SCENE_ANALYZER_URL || 'http://localhost:8000';
const MUSIC_GENERATOR_URL = process.env.MUSIC_GENERATOR_URL || 'http://localhost:8001';

// Middleware configuration
app.use(cors());
app.use(express.json());

/**
 * API Routes
 */
app.post('/api/scene', async (req: Request, res: Response) => {
  const { scene } = req.body;
  
  try {
    // Send scene to analyzer
    const analysisResponse = await fetch(`${SCENE_ANALYZER_URL}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: scene })
    });
    
    const analysis = await analysisResponse.json();
    
    // Generate music based on analysis
    const musicResponse = await fetch(`${MUSIC_GENERATOR_URL}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(analysis)
    });
    
    res.json(await musicResponse.json());
  } catch (error) {
    console.error('Error processing scene:', error);
    res.status(500).json({ error: 'Failed to process scene' });
  }
});

// Serve static files from the React application
app.use(express.static(path.join(__dirname, '../../dist')));

// Basic error handling middleware
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  console.error(err.stack);
  res.status(500).send('Something broke!');
});

// Handle React routing, return all requests to React application
app.get('*', (req: Request, res: Response) => {
  res.sendFile(path.join(__dirname, '../../dist/index.html'));
});

app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
