import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './App/index.css'
import './index.css'
import App from './App/index.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
