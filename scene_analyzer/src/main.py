import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, List
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI application
app = FastAPI(title="Scene Analyzer Service")

# Configure CORS middleware to allow cross-origin requests
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

"""
 Data models for request/response handling
"""
class SceneInput(BaseModel):
    text: str

class SceneAnalysis(BaseModel):
    mood: str
    key_elements: List[str]
    suggested_music_type: str

"""
 Core analysis functions
"""
def analyze_text(text: str) -> Dict:
    """
    Analyzes scene text to determine mood, key elements, and suggested music type
    Returns a dictionary containing analysis results
    """
    text_lower = text.lower()

    # Basic mood detection
    moods = {
        "combat": ["battle", "fight", "attack", "clash"],
        "tense": ["dark", "danger", "scary", "mysterious"],
        "peaceful": ["quiet", "calm", "gentle", "serene"],
        "epic": ["legendary", "massive", "powerful", "grand"]
    }

    # Find mood based on keywords
    scene_mood = "neutral"
    for mood, keywords in moods.items():
        if any(word in text_lower for word in keywords):
            scene_mood = mood
            break

    # Extract key elements (simple keyword spotting)
    key_elements = []
    important_elements = [
        "dragon", "sword", "magic", "forest", "dungeon",
        "tavern", "castle", "monster", "treasure"
    ]

    for element in important_elements:
        if element in text_lower:
            key_elements.append(element)

    # Determine music type based on mood and elements
    music_types = {
        "combat": "battle",
        "tense": "suspense",
        "peaceful": "ambient",
        "epic": "orchestral"
    }

    music_type = music_types.get(scene_mood, "ambient")

    return {
        "mood": scene_mood,
        "key_elements": key_elements or ["general scene"],
        "suggested_music_type": music_type
    }

@app.post("/analyze", response_model=SceneAnalysis)
async def analyze_scene(scene: SceneInput) -> Dict:
    """
    Analyze the given scene text and return analysis results.
    """
    return analyze_text(scene.text)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
