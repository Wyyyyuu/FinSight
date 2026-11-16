import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import './index.css'
import App from './App.tsx'
import { ErrorBoundary } from './components/ui/ErrorBoundary.tsx'
import { ensurePublicDemoSession } from './api/publicDemoSession'

const renderApp = () => createRoot(document.getElementById('root')!).render(
  <ErrorBoundary>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </ErrorBoundary>
)

// Render the welcome page while the API client establishes the visitor cookie.
void ensurePublicDemoSession()
renderApp()
