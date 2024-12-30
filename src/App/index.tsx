import './index.css'
import PromptInput from "./PromptInput";
import MediaPlayer from "./MediaPlayer";
import { useState } from 'react';

interface AudioData {
    audioUrl: string | null;
    duration: number;
}

function App() {
    const [audioData, setAudioData] = useState<AudioData>({
        audioUrl: null,
        duration: 0
    });

    return (
        <div className="app-container">
            <h1 className="title">Bardic Companion</h1>
            <MediaPlayer audioData={audioData} />
            <PromptInput onAudioGenerated={setAudioData} />
        </div>
    )
}

export default App
