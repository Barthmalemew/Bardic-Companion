from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple
import mingus.midi as midi
from mingus.containers import Note, NoteContainer, Track, Composition
from fastapi.middleware.cors import CORSMiddleware
from midiutil import MIDIFile
import numpy as np
import soundfile as soundfile
from pydub import AudioSegment
import io
import os
from pathlib import Path

# Initialize FastAPI application
app = FastAPI(title="Music Generator Service")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Define request/response models
class MusicRequest(BaseModel):
    mood: str
    setting: str
    intensity: float

class MusicResponse(BaseModel):
    audio_url: str
    duration: float

# Musical configuration constants
SCALES = {
    "peaceful": [60, 62, 64, 65, 67, 69, 71, 72],  # C major
    "tense": [60, 61, 64, 65, 67, 68, 71, 72],     # C minor
    "epic": [60, 62, 64, 67, 69, 72, 74, 76],      # C major pentatonic extended
    "combat": [60, 63, 65, 66, 68, 70, 72, 73]     # C diminished
}

BASE_DIR = Path(__file__).parent
LOOPS_DIR = BASE_DIR / "base_loops"

AMBIENT_LOOPS = {
    "dungeon": str(LOOPS_DIR / "dungeon_background.wav"),
    "forest": str(LOOPS_DIR / "forest_background.wav"),
    "town": str(LOOPS_DIR / "town_background.wav"),
    "combat": str(LOOPS_DIR / "battle_background.wav")
}

class AudioProcessor:
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.base_path = str(LOOPS_DIR)
        self.temp_path = LOOPS_DIR / "temp"  
        self.temp_path.mkdir(exist_ok=True)

    def get_temp_path(self, filename: str) -> str:
        """Generate a path for temporary files within our project structure"""
        return str(self.temp_path / filename)

    def cleanup_temp_files(self):
        """Remove temporary files after processing"""
        for temp_file in self.temp_path.glob("*"):
            try:
                temp_file.unlink()
            except Exception as e:
                print(f"Failed to remove temporary file {temp_file}: {e}")

    def load_audio_file(self, filepath: str) -> Optional[AudioSegment]:
        """Load an audio file and return as AudioSegment"""
        try:
            return AudioSegment.from_file(filepath)
        except Exception as exception:
            print(f"Error loading audio file: {exception}")
            return None

    def midi_to_audio(self, midi_path: str) -> str:
        """Convert MIDI file to audio using FluidSynth"""
        output_path = self.get_temp_path("synthesized.wav")
        
        try:
            # Convert MIDI to audio using system FluidSynth
            os.system(f'fluidsynth -ni /usr/share/sounds/sf2/FluidR3_GM.sf2 {midi_path} -F {output_path} -r 44100')
            if not os.path.exists(output_path):
                raise ValueError("Failed to generate audio file")
            return output_path
        except Exception as e:
            raise ValueError(f"Failed to synthesize MIDI: {e}")

    def combine_tracks(self,
                       tracks: list[AudioSegment],
                       volumes: list[float]) -> AudioSegment:
        """Combine multiple audio tracks with respective volumes"""
        if not tracks:
            raise ValueError("No tracks provided for mixing")

        # Normalize all tracks to same length
        max_length = max(len(track) for track in tracks)
        normalized_tracks = [
            self._extend_track(track, max_length) for track in tracks
        ]

        # Apply volumes and mix
        result = None
        for track, volume in zip(normalized_tracks, volumes):
            adjusted = track - (20 - (volume * 20))  # Convert 0-1 to dB scale
            if result is None:
                result = adjusted
            else:
                result = result.overlay(adjusted)

        return result

    def _extend_track(self, track: AudioSegment, target_length: int) -> AudioSegment:
        """Loop track to match target length"""
        if len(track) >= target_length:
            return track[:target_length]

        repeats = (target_length // len(track)) + 1
        return track * repeats

    def export_audio(self,
                     audio: AudioSegment,
                     format: str = 'wav') -> Tuple[bytes, float]:
        """Export audio to bytes and return duration"""
        buffer = io.BytesIO()
        audio.export(buffer, format=format)
        return buffer.getvalue(), len(audio) / 1000.0  # Duration in seconds

# Initialize audio processor
audio_processor = AudioProcessor()

def generate_midi(mood: str, intensity: float, duration: int = 30) -> str:
    """
    Generate MIDI music based on mood and intensity.
    Uses predefined scales and adjusts musical parameters based on intensity.

    Args:
        mood: The emotional mood (peaceful, tense, epic, combat)
        intensity: Float between 0-1 indicating the intensity level
        duration: Length of the music in seconds

    Returns:
        str: Path to the generated MIDI file
    """
    # Create a single-track MIDI file (to match your existing mixing setup)
    midi = MIDIFile(1)
    track = 0
    time = 0

    # Set tempo based on intensity
    base_tempo = 90
    tempo = int(base_tempo + (intensity * 60))  # Varies from 90-150 BPM
    midi.addTempo(track, time, tempo)

    # Get the appropriate scale from your existing SCALES dictionary
    scale = SCALES.get(mood, SCALES["peaceful"])

    # Set up musical parameters based on intensity
    volume = int(60 + (intensity * 40))  # Volume ranges from 60-100

    # Determine note duration range based on intensity
    if intensity < 0.3:
        note_durations = [1.0, 1.5]      # Longer notes for calm scenes
    elif intensity < 0.7:
        note_durations = [0.5, 1.0]      # Medium notes for moderate intensity
    else:
        note_durations = [0.25, 0.5]     # Shorter notes for high intensity

    current_time = 0
    while current_time < duration:
        # Choose a note from the scale, favoring stepwise motion
        note = scale[np.random.randint(0, len(scale))]

        # Select note duration from our intensity-based options
        note_duration = np.random.choice(note_durations)

        # Add the note to our MIDI file
        midi.addNote(track, 0, note, current_time, note_duration, volume)

        # Move to the next time position
        current_time += note_duration

    # Save the MIDI file to your temp location
    output_path = "temp_melody.mid"
    with open(output_path, "wb") as midi_file:
        midi.writeFile(midi_file)

    return output_path

def get_ambient_track(setting: str, duration: int = 30) -> Optional[AudioSegment]:
    """Get the appropriate ambient background track and loop it"""
    loop_path = AMBIENT_LOOPS.get(setting, AMBIENT_LOOPS["town"])
    track = audio_processor.load_audio_file(loop_path)
    if track:
        return audio_processor._extend_track(track, duration * 1000)
    return None

def mix_tracks(midi_path: str, ambient_track: AudioSegment, intensity: float) -> bytes:
    """Mix the MIDI melody with the ambient background"""
    # Convert MIDI to audio and load it
    wav_path = audio_processor.midi_to_audio(midi_path)
    melody = audio_processor.load_audio_file(wav_path)
    if not melody or not ambient_track:
        raise ValueError("Failed to load audio tracks")

    mixed = audio_processor.combine_tracks(
        [melody, ambient_track],
        [intensity, 1.0 - (intensity * 0.5)]
    )
    return audio_processor.export_audio(mixed)[0]

@app.post("/generate", response_model=MusicResponse)
async def generate_music(request: MusicRequest) -> Dict:
    """
    Generate hybrid music combining MIDI and ambient tracks.
    """
    try:
        # Generate MIDI
        midi_file = generate_midi(
            mood=request.mood,
            intensity=request.intensity
        )

        # Get ambient background
        ambient_track = get_ambient_track(
            setting=request.setting,
            duration=30
        )

        # Mix tracks
        final_mix = mix_tracks(
            midi_file,
            ambient_track,
            request.intensity
        )

        return {
            "audio_url": f"/audio/{final_mix}",
            "duration": 30.0
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary files
        audio_processor.cleanup_temp_files()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
