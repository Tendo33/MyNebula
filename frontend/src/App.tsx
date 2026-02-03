import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Settings from './pages/Settings';




function App() {
  return (
    <Router>
      <div className="min-h-screen bg-nebula-bg text-nebula-text-main font-sans selection:bg-nebula-primary selection:text-nebula-bg">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/graph" element={<Dashboard />} />
          <Route path="/data" element={<Dashboard />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>

      </div>
    </Router>
  );
}

export default App;
