import './index.css'
import './PromptInput/index.tsx'
import PromptInput from "./PromptInput";
import './MediaPlayer/index.tsx'
import MediaPlayer from "./MediaPlayer";

/**
 * Main App component - Orchestrates the application layout and components
 */
function App() {
    return (
        <div className="app-container">
            <h1 className="title">Bardic Companion</h1>
            <MediaPlayer/>
            <PromptInput/>
        </div>
    )
}

export default App
