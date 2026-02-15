import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import { GraphProvider, useGraph } from './contexts/GraphContext';
import { AdminAuthProvider } from './contexts/AdminAuthContext';
import Dashboard from './pages/Dashboard';
import GraphPage from './pages/GraphPage';
import DataPage from './pages/DataPage';
import Settings from './pages/Settings';
import CommandPalette from './components/ui/CommandPalette';
import useCommandPalette from './hooks/useCommandPalette';

// Inner component that uses router hooks
function AppContent() {
  const navigate = useNavigate();
  const { isOpen, close } = useCommandPalette();
  const { setSelectedNode } = useGraph();

  const handleSelectNode = (node: any) => {
    setSelectedNode(node);
    navigate('/graph');
  };

  const handleSelectCluster = (cluster: any) => {
    navigate(`/graph?cluster=${cluster.id}`);
  };

  return (
    <>
      <div className="min-h-screen bg-bg-main text-text-main font-sans selection:bg-action-primary/20">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/graph" element={<GraphPage />} />
          <Route path="/data" element={<DataPage />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </div>

      {/* Global Command Palette (Cmd/Ctrl + K) */}
      <CommandPalette
        isOpen={isOpen}
        onClose={close}
        onSelectNode={handleSelectNode}
        onSelectCluster={handleSelectCluster}
      />
    </>
  );
}

function App() {
  return (
    <GraphProvider>
      <AdminAuthProvider>
        <Router>
          <AppContent />
        </Router>
      </AdminAuthProvider>
    </GraphProvider>
  );
}

export default App;
