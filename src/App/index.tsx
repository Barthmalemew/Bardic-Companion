import './index.css'
import './PromptInput/index.tsx'
import PromptInput from "./PromptInput";
import './MediaPlayer/index.tsx'
import MediaPlayer from "./MediaPlayer";

function App() {
    return (
        <div className="app-container">
            <h1>Bardic Companion</h1>
            <MediaPlayer/>
            <PromptInput/>
        </div>
    )
}

export default App
