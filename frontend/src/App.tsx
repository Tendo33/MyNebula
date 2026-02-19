import { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate } from 'react-router-dom';
import { GraphProvider, useGraph } from './contexts/GraphContext';
import { AdminAuthProvider } from './contexts/AdminAuthContext';
import CommandPalette from './components/ui/CommandPalette';
import useCommandPalette from './hooks/useCommandPalette';
import { ClusterInfo, GraphNode } from './types';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const GraphPage = lazy(() => import('./pages/GraphPage'));
const DataPage = lazy(() => import('./pages/DataPage'));
const Settings = lazy(() => import('./pages/Settings'));

// Inner component that uses router hooks
function AppContent() {
  const navigate = useNavigate();
  const { isOpen, close } = useCommandPalette();
  const { setSelectedNode } = useGraph();

  const handleSelectNode = (node: GraphNode) => {
    setSelectedNode(node);
    navigate('/graph');
  };

  const handleSelectCluster = (cluster: ClusterInfo) => {
    navigate(`/graph?cluster=${cluster.id}`);
  };

  return (
    <>
      <div className="min-h-screen bg-bg-main text-text-main font-sans selection:bg-action-primary/20">
        <Suspense fallback={<div className="flex min-h-screen items-center justify-center text-sm text-text-muted">Loading...</div>}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/graph" element={<GraphPage />} />
            <Route path="/data" element={<DataPage />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </Suspense>
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
