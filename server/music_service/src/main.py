from contextlib import asynccontextmanager
from urllib.request import Request
import logger
import uvicorn
import math
import random
from fastapi import FastAPI, HTTPException
import logging
from typing import Dict, Optional
import base64
import numpy as np
from pydantic import BaseModel
from pydub import AudioSegment
from pydub.generators import Sine, Square, Triangle, Sawtooth
import io
from mingus.core import chords, progressions, scales
from mingus.containers import Note, NoteContainer, Track, Composition
from scipy import signal
from starlette.middleware.cors import CORSMiddleware


# Define AudioProcessor class with all necessary methods
class AudioProcessor:
    def __init__(self, sample_rate: int = 44100):
        """Initialize AudioProcessor with audio generation parameters"""
        self.sample_rate = sample_rate
        self.current_mood = None
        self.duration = 30.0
        self.composition = Composition()
        self.current_track = Track()
        # Add debugging for audio synthesis
        self.debug = True
        logger.debug("Initializing AudioProcessor with sample rate: %d", sample_rate)
        # Define musical parameters
        self.note_map = {
            'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11
        }
        # Add ADSR envelope parameters for each mood
        self.mood_settings = {
            'peaceful': {
                'tempo': 80,
                'key': 'C',
                'scale': scales.Major,
                'progression': [["C", "Am", "F", "G"], ["C", "F", "G", "C"]],
                'waveform': 'sine',
                'attack': 0.1,
                'decay': 0.2,
                'sustain': 0.7,
                'release': 0.3
            },
            'combat': {
                'tempo': 140,
                'key': 'Em',
                'scale': scales.Phrygian,
                'progression': [["Em", "C", "G", "B"], ["Em", "D", "C", "B"]],
                'waveform': 'sawtooth',
                'attack': 0.05,
                'decay': 0.1,
                'sustain': 0.8,
                'release': 0.1
            }
        }
        # Define base frequencies for note generation
        self.base_freq = 440.0  # A4 = 440Hz
        self.note_frequencies = {
            'C': 261.63, 'D': 293.66, 'E': 329.63,
            'F': 349.23, 'G': 392.00, 'A': 440.00,
            'B': 493.88
        }
        self.waveform_generators = {
            'sine': Sine,
            'square': Square,
            'triangle': Triangle,
            'sawtooth': Sawtooth
        }
        self.scale_types = {
            "major": scales.Major,
            "minor": scales.NaturalMinor,
            "phrygian": scales.Phrygian,
            "dorian": scales.Dorian,
            "mixolydian": scales.Mixolydian
        }

    def generate_audio(self, mood: str, duration: float = 30.0) -> bytes:
        """Generate audio based on mood and duration"""
        try:
            self.current_mood = mood
            self.duration = duration
            params = self.mood_settings.get(mood, self.mood_settings["peaceful"])

            # Reset composition and tracks
            self.composition = Composition()
            self.current_track = Track()

            # Create musical elements based on mood
            logger.debug(f"Generating {mood} music with params: {params}")
            self.create_chord_progression(params["progression"], params["tempo"])
            self.add_melody(params["scale"], params["key"])

            # Generate and process audio
            audio_data = self._composition_to_audio()
            processed_audio = self.apply_effects(audio_data)

            # Convert to WAV format
            return self._convert_to_wav(processed_audio)
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            raise

    def create_chord_progression(self, progression, tempo):
        """Create chord progression from given parameters"""
        try:
            logger.debug(f"Creating chord progression: {progression} at tempo {tempo}")
            for chord_sequence in progression:
                for chord_name in chord_sequence:
                    logger.debug(f"Processing chord: {chord_name}")
                    chord_notes = chords.from_shorthand(chord_name)
                    if not chord_notes:
                        raise ValueError(f"Invalid chord: {chord_name}")
                    logger.debug(f"Generated notes for chord {chord_name}: {chord_notes}")
                    self.current_track.add_notes(chord_notes, duration=2)
            self.composition.add_track(self.current_track)
            logger.debug(f"Added chord progression track with {len(self.current_track)} notes")
        except Exception as e:
            logger.error(f"Error in chord progression: {e}")
            raise

    def add_melody(self, scale_name, key):
        try:
            # Extract root note and mode from key (e.g., 'Em' -> 'E', 'minor')
            root = key[0].upper()
            mode = 'minor' if 'm' in key else 'major'
            
            # Create proper scale
            scale = scales.Major(root) if mode == 'major' else scales.NaturalMinor(root)
            scale_notes = scale.ascending()
            
            logger.debug(f"Generated scale notes for {key} {mode}: {scale_notes}")
            melody = self._generate_melody(scale_notes)
            melody_track = Track()
            for note in melody:
                melody_track.add_notes(note, duration=1)
            self.composition.add_track(melody_track)
        except KeyError:
            logger.error(f"Invalid scale name: {scale_name}")
            raise ValueError(f"Unsupported scale type: {scale_name}")
        except Exception as e:
            logger.error(f"Error adding melody: {e}")
            raise

    def _composition_to_audio(self) -> np.ndarray:
        """Convert musical composition to audio samples"""
        try:
            samples = np.zeros(int(self.duration * self.sample_rate))
            logger.debug(f"Created initial audio buffer: {len(samples)} samples")
            
            for track in self.composition:
                track_samples = np.zeros_like(samples)
                current_position = 0

                logger.debug(f"Processing track with {len(track)} notes")
                for note in track:
                    if isinstance(note, Note):
                        freq = 440.0 * (2.0 ** ((note.int() - 69) / 12.0))
                        logger.debug(f"Generating audio for note {note.name} at {freq}Hz")
                        
                        duration = 0.5  # half second per note
                        samples_per_note = int(duration * self.sample_rate)
                        t = np.linspace(0, duration, samples_per_note)
                        
                        # Generate actual waveform with harmonics
                        wave = np.zeros_like(t)
                        for harmonic in range(1, 4):
                            wave += (1.0 / harmonic) * np.sin(2 * np.pi * freq * harmonic * t)
                        
                        # Normalize wave
                        wave = wave / np.max(np.abs(wave))

                        # Apply ADSR envelope
                        mood_params = self.mood_settings[self.current_mood]
                        envelope = self._create_adsr_envelope(
                            samples_per_note,
                            mood_params['attack'],
                            mood_params['decay'],
                            mood_params['sustain'],
                            mood_params['release']
                        )
                        wave *= envelope
                        wave *= 0.5  # Reduce amplitude to prevent clipping

                        # Add to track samples
                        end_pos = min(len(samples), current_position + len(wave))
                        track_samples[current_position:end_pos] += wave[:end_pos-current_position]
                        current_position += len(wave)

                # Mix track into main samples
                samples += track_samples * 0.5
            return samples
        except Exception as e:
            logger.error(f"Error in composition to audio conversion: {e}")
            raise

    def _generate_melody(self, scale_notes):
        """Generate melodic pattern based on scale notes"""
        try:
            logger.debug(f"Generating melody from scale notes: {scale_notes}")
            melody = []
            # Create more musical phrases
            for _ in range(8):  # 8 bars
                note = random.choice(scale_notes)
                # Add some rhythm variation
                duration = random.choice([0.5, 1.0, 1.5])
                melody.append(Note(note))
                
            logger.debug(f"Generated melody with {len(melody)} notes")
            return melody
        except Exception as e:
            logger.error(f"Error generating melody: {e}")
            raise

    def apply_effects(self, audio_data):
        try:
            # Add reverb using proper scipy window function
            impulse_response = signal.windows.hann(1000)
            audio_with_reverb = signal.convolve(audio_data, impulse_response, mode='same')
            
            # Normalize
            audio_normalized = audio_with_reverb / np.max(np.abs(audio_with_reverb))
            
            return audio_normalized
        except Exception as e:
            logger.error(f"Error applying audio effects: {e}")
            raise

    def _convert_to_wav(self, processed_audio) -> bytes:
        """Convert processed audio to WAV format"""
        try:
            # Verify audio data before conversion
            if np.all(processed_audio == 0):
                logger.error("Audio data is empty (all zeros)")
                raise ValueError("Generated audio contains no sound data")

            logger.debug(f"Audio stats before conversion - min: {np.min(processed_audio)}, " +
                      f"max: {np.max(processed_audio)}, mean: {np.mean(processed_audio)}")

            # Normalize audio
            max_val = np.max(np.abs(processed_audio))
            if max_val > 0:
                processed_audio = processed_audio / max_val
            
            # Ensure we have non-zero audio data
            if np.all(processed_audio == 0):
                raise ValueError("Audio normalization resulted in silence")
            
            processed_audio = np.int16(processed_audio * 32767)
            
            # Create stereo audio
            stereo_audio = np.column_stack((processed_audio, processed_audio))
            
            audio_segment = AudioSegment(
                stereo_audio.tobytes(),
                frame_rate=self.sample_rate,
                sample_width=2,
                channels=2
            )

            buffer = io.BytesIO()
            audio_segment.export(buffer, format="wav")
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Error converting to WAV: {e}")
            raise

    def _create_adsr_envelope(self, samples_per_note, attack, decay, sustain, release):
        """Create an ADSR envelope"""
        attack_samples = int(attack * self.sample_rate)
        decay_samples = int(decay * self.sample_rate)
        sustain_samples = samples_per_note - (attack_samples + decay_samples + int(release * self.sample_rate))
        release_samples = int(release * self.sample_rate)

        envelope = np.zeros(samples_per_note)
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain, decay_samples)
        envelope[attack_samples + decay_samples:attack_samples + decay_samples + sustain_samples] = sustain
        envelope[attack_samples + decay_samples + sustain_samples:] = np.linspace(sustain, 0, release_samples)

        return envelope

    def _note_to_midi(self, note):
        """Convert a Note object to MIDI number"""
        return self.note_map[note.name[0]] + (12 * (note.octave + 1))

# Global AudioProcessor instance
audio_processor = None

def init_audio_processor() -> AudioProcessor:
    """Initialize and configure the AudioProcessor instance"""
    try:
        logger.info("Initializing AudioProcessor...")
        processor = AudioProcessor()
        
        # Verify the processor is working by generating a test tone
        test_audio = processor.generate_audio("peaceful", duration=1.0)
        if not test_audio or len(test_audio) == 0:
            raise RuntimeError("AudioProcessor failed to generate test audio")
            
        logger.info("AudioProcessor initialized successfully")
        return processor
    except Exception as e:
        logger.error(f"Failed to initialize AudioProcessor: {e}")
        raise RuntimeError(f"AudioProcessor initialization failed: {str(e)}")

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global audio_processor
    logger.info("Starting up music service...")
    audio_processor = init_audio_processor()
    yield
    # Shutdown
    logger.info("Shutting down music service...")

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('music_service.log')
    ]
)
logger = logging.getLogger(__name__)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response: Status {response.status_code}")
    return response

class SceneInput(BaseModel):
    text: str

class MusicResponse(BaseModel):
    audio_url: str
    duration: float

# Basic mood detection patterns
MOOD_PATTERNS = {
    "combat": {
        "tempo": 140,
        "key": "C",
        "scale": "Phrygian",
        "progression": [["Cm", "G#", "Fm", "G"], ["Cm", "Dm", "Em", "G"]],
        "rhythm": "aggressive"
    },
    "peaceful": {
        "tempo": 80,
        "key": "G",
        "scale": "Major",
        "progression": [["G", "Em", "C", "D"], ["G", "C", "D", "G"]],
        "rhythm": "flowing"
    }
    # Add more moods here
}

def analyze_scene(text: str) -> Dict:
    """Analyze scene text to determine appropriate musical mood"""
    text_lower = text.lower()
    
    # Define mood keywords
    mood_keywords = {
        'combat': ['battle', 'fight', 'combat', 'war', 'attack', 'danger'],
        'peaceful': ['forest', 'peaceful', 'calm', 'serene', 'quiet', 'gentle'],
        'mysterious': ['cave', 'dungeon', 'dark', 'mysterious', 'strange'],
        'celebratory': ['tavern', 'celebration', 'party', 'festival', 'happy']
    }
    
    # Count keyword matches for each mood
    mood_scores = {mood: 0 for mood in mood_keywords}
    for mood, keywords in mood_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                mood_scores[mood] += 1
    
    # Select mood with highest score, default to peaceful if no matches
    selected_mood = max(mood_scores.items(), key=lambda x: x[1])[0]
    logger.debug(f"Scene analysis: {text_lower} -> {selected_mood} (scores: {mood_scores})")
    
    return {"mood": selected_mood if mood_scores[selected_mood] > 0 else "peaceful"}

@app.post("/process")
async def process_scene(scene: SceneInput) -> MusicResponse:
    try:
        global audio_processor
        logger.debug(f"Processing scene with text: {scene.text}")
        if audio_processor is None:
            audio_processor = init_audio_processor()
            
        logger.debug(f"Received scene text: {scene.text}")
        
        # Validate input
        if not scene.text.strip():
            raise HTTPException(status_code=400, detail="Empty scene text provided")
        
        analysis = analyze_scene(scene.text)
        logger.debug(f"Scene analysis result: {analysis}")
        
        logger.debug("Generating audio...")
        audio_bytes = audio_processor.generate_audio(analysis["mood"])
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=500, detail="Generated audio is empty")
        
        logger.info(f"Successfully generated audio: {len(audio_bytes)} bytes, mood: {analysis['mood']}")
        
        # Convert audio bytes to base64 for direct playback
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        audio_url = f"data:audio/wav;base64,{audio_base64}"
        logger.debug("Audio conversion complete, sending response")
        return MusicResponse(
            audio_url=audio_url,
            duration=30.0
        )
    except Exception as exception:
        logger.error(f"Error processing scene: {str(exception)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exception))

@app.get("/health")
async def health_check():
    try:
        global audio_processor
        if audio_processor is None:
            audio_processor = init_audio_processor()
        # Verify AudioProcessor is working
        test_audio = audio_processor.generate_audio("peaceful", duration=1.0)
        return {"status": "healthy", "audioProcessor": "operational"}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
