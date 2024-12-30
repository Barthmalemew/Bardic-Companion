import express, { Request, Response, NextFunction } from 'express';
import path from 'path';
import dotenv from 'dotenv';
import winston from 'winston';
import debug from 'debug';
import cors from 'cors';
// Load environment variables
dotenv.config();

// Initialize Express application
const app = express();
const PORT = process.env.PORT || 3000;

// Music Service URL
const MUSIC_SERVICE_URL = process.env.MUSIC_SERVICE_URL || 'http://localhost:8000';

// Enhanced logging setup
const logger = winston.createLogger({
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.json(),
    winston.format.prettyPrint()
  ),
  transports: [
    new winston.transports.Console({
      format: winston.format.simple()
    }),
    new winston.transports.File({ 
      filename: 'logs/error.log', 
      level: 'error' 
    })
  ]
});

// Middleware configuration
app.use(cors());
app.use(express.json());

// Configure headers middleware
app.use((req, res, next) => {
  res.header('Content-Type', 'application/json');
  next();
});

const log = debug('bardic:server');

/**
 * API Routes
 */
app.post('/api/scene', async (req: Request, res: Response) => {
  const { scene } = req.body;
  
  // Validate scene input
  if (!scene || typeof scene !== 'string') {
    logger.error('Invalid scene input received');
    return res.status(400).json({ 
      error: 'Invalid input', 
      details: 'Scene text is required and must be a string' 
    });
  }
  logger.info(`Processing scene request: ${scene.substring(0, 100)}...`);

  // Add content type validation header
  const headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json'
  };

  log(`Received scene request: ${scene}`);
  try {
    // Send scene to music service
    log(`Sending request to music service at ${MUSIC_SERVICE_URL}`);
    const response = await fetch(`${MUSIC_SERVICE_URL}/process`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ text: scene })
    });

    if (!response.ok) {
      throw new Error(`Music service responded with status: ${response.status}`);
    }

    // Log response headers for debugging
    log(`Response headers: ${JSON.stringify([...response.headers])}`);

    const data = await response.json();
    log(`Received response from music service: ${JSON.stringify(data)}`);
    res.json(data);
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
