import { useEffect, useState } from 'react';
import './index.css';

interface AudioState {
    url: string | null;
    isPlaying: boolean;
}

function MediaPlayer() {
    const [audioState, setAudioState] = useState<AudioState>({
        url: null,
        isPlaying: false
    });

    // Audio element reference
    const audioRef = useState<HTMLAudioElement | null>(null);

    useEffect(() => {
        // Initialize audio element
        if (!audioRef.current) {
            audioRef.current = new Audio();
        }
    }, []);

    return (
        <div className="media-player-container">
            <div>Media Player Component</div>
            {/* Add audio controls here later */}
        </div>
    );
}

export default MediaPlayer;
