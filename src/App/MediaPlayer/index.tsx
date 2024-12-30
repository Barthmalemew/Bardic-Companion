import { useEffect, useRef, useState } from 'react';
import './index.css';

// Enhanced error types
interface AudioError {
    code: number;
    message: string;
    timestamp: Date;
    details?: string;
}

interface MediaPlayerProps {
    audioData: {
        audioUrl: string | null;
        duration: number;
    };
}

// Add debug logging utility
const debug = (message: string, data?: any) => {
    if (import.meta.env.DEV) {
        console.log(`[MediaPlayer] ${message}`, data || '');
    }
};

interface AudioValidationResult {
    isValid: boolean;
    error?: string;
    processedUrl?: string;
}

function MediaPlayer({ audioData }: MediaPlayerProps) {
    const audioRef = useRef<HTMLAudioElement | null>(null);
    const [error, setError] = useState<AudioError | null>(null);
    
    // Add volume controls and debugging
    useEffect(() => {
        const audio = audioRef.current;
        if (audio) {
            audio.volume = 0.5; // Set initial volume
            audio.addEventListener('volumechange', () => {
                console.log(`Volume changed to: ${audio.volume}`);
            });
            audio.addEventListener('playing', () => {
                console.log('Audio started playing:', { currentTime: audio.currentTime, duration: audio.duration });
            });
        }
    }, [audioRef.current]);

    const validateAndProcessAudio = (audioData: string): AudioValidationResult => {
        // Validate data URL format
        if (!audioData) {
            debug('No audio data provided');
            return {
                isValid: false,
                error: 'No audio data provided'
            };
        }
        debug('Validating audio data', audioData.substring(0, 50));

        // Check if it's already a proper data URL
        if (audioData.startsWith('data:audio/wav;base64,')) {
            return {
                isValid: true,
                processedUrl: audioData
            };
        }

        // Try to process raw base64 string
        try {
            return {
                isValid: true,
                processedUrl: `data:audio/wav;base64,${audioData}`
            };
        } catch (err) {
            console.error('Error processing audio:', err);
            setError({
                code: 1,
                message: 'Invalid audio data received',
                timestamp: new Date(),
                details: err instanceof Error ? err.message : 'Unknown error'
            });
            return {
                isValid: false,
                error: 'Invalid audio format'
            };
        }
    };

    const handleAudioError = (event: Event) => {
        const target = event.target as HTMLAudioElement;
        console.error('Audio error:', {
            error: target.error,
            currentSrc: target.currentSrc,
            readyState: target.readyState,
            networkState: target.networkState
        });
        setError({ code: 3, message: 'Failed to play audio. Please try again.', timestamp: new Date() });
    };

    useEffect(() => {
        setError(null);
        
        if (!audioData?.audioUrl) return;

        const validationResult = validateAndProcessAudio(audioData.audioUrl);
        if (!validationResult.isValid) {
            setError({ code: 2, message: validationResult.error || 'Invalid audio format', timestamp: new Date() });
            return;
        }

        const audio = audioRef.current;
        if (!audio) return;

        // Set the processed URL
        audio.src = validationResult.processedUrl || '';

        // Add event listeners
        const handleCanPlay = () => {
            audio.play().catch(err => {
                console.error('Playback failed:', err);
                setError({ code: 4, message: 'Failed to start playback', timestamp: new Date() });
            });
        };

        audio.addEventListener('error', handleAudioError);
        audio.addEventListener('canplay', handleCanPlay);

        return () => {
            audio.removeEventListener('error', handleAudioError);
            audio.removeEventListener('canplay', handleCanPlay);
        };
    }, [audioData.audioUrl]);

    useEffect(() => {
        return () => {
            if (audioRef.current) {
                audioRef.current.pause();
            }
        };
    }, []);

    return (
        <div className="media-player-container">
            <div className="player-content">
                {error && (
                    <div className="error-message">
                        <p>Error: {error.message}</p>
                        {error.details && (
                            <p className="error-details">
                                Details: {error.details}
                            </p>
                        )}
                    </div>
                )}
                {audioData.audioUrl ? (
                    <div>
                        <p>Now Playing</p>
                        <audio
                            controls
                            ref={audioRef}
                            onError={(e) => console.error('Audio error:', e)}
                            preload="auto"
                        />
                    </div>
                ) : (
                    <p>Enter a scene description to generate music</p>
                )}
            </div>
        </div>
    );
}

export default MediaPlayer;
