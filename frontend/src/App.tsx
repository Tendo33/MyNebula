import { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation, useNavigate } from 'react-router-dom';
import { ErrorBoundary } from 'react-error-boundary';
import { useTranslation } from 'react-i18next';
import { GraphProvider, useGraph } from './contexts/GraphContext';
import { AdminAuthProvider } from './contexts/AdminAuthContext';
import { ErrorFallback } from './components/ui/ErrorFallback';
import CommandPalette from './components/ui/CommandPalette';
import useCommandPalette from './hooks/useCommandPalette';
import { ClusterInfo, GraphNode } from './types';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const GraphPage = lazy(() => import('./pages/GraphPage'));
const DataPage = lazy(() => import('./pages/DataPage'));
const Settings = lazy(() => import('./pages/Settings'));

// Inner component that uses router hooks
function GraphAppContent({
  isOpen,
  close,
}: {
  isOpen: boolean;
  close: () => void;
}) {
  const navigate = useNavigate();
  const { setSelectedNode } = useGraph();

  const handleSelectNode = (node: GraphNode) => {
    setSelectedNode(node);
    navigate(`/graph?node=${node.id}`);
  };

  const handleSelectCluster = (cluster: ClusterInfo) => {
    navigate(`/graph?cluster=${cluster.id}`);
  };

  const handleSelectSearch = (
    value: string,
    facet: 'search' | 'language' | 'tag' = 'search'
  ) => {
    const params = new URLSearchParams();
    if (facet === 'language') params.set('language', value);
    else if (facet === 'tag') params.set('tag', value);
    else params.set('q', value);
    navigate(`/graph?${params.toString()}`);
  };

  const { t } = useTranslation();

  return (
    <>
      <div className="min-h-screen bg-bg-main text-text-main dark:bg-dark-bg-main dark:text-dark-text-main font-sans selection:bg-action-primary/20">
        <ErrorBoundary FallbackComponent={ErrorFallback}>
          <Suspense fallback={
            <div className="flex min-h-screen items-center justify-center text-sm text-text-muted">
              {t('common.loading', 'Loading...')}
            </div>
          }>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/graph" element={<GraphPage />} />
              <Route path="/data" element={<DataPage />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </Suspense>
        </ErrorBoundary>
      </div>

      <CommandPalette
        isOpen={isOpen}
        onClose={close}
        onSelectNode={handleSelectNode}
        onSelectCluster={handleSelectCluster}
        onSelectSearch={handleSelectSearch}
      />
    </>
  );
}

function AppContent() {
  const location = useLocation();
  const { isOpen, close } = useCommandPalette();
  const graphEnabled = location.pathname === '/graph';

  return (
    <GraphProvider enabled={graphEnabled}>
      <GraphAppContent isOpen={isOpen} close={close} />
    </GraphProvider>
  );
}

function App() {
  return (
    <AdminAuthProvider>
      <Router future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
        <AppContent />
      </Router>
    </AdminAuthProvider>
  );
}

export default App;
