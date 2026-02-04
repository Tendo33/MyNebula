import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import GraphPage from './pages/GraphPage';
import DataPage from './pages/DataPage';
import Settings from './pages/Settings';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-bg-main text-text-main font-sans selection:bg-action-primary/20">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/graph" element={<GraphPage />} />
          <Route path="/data" element={<DataPage />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
