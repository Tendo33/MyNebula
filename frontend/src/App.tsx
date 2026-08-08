import { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Link, Routes, Route, useLocation, useNavigate } from 'react-router-dom';
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

const NotFound = () => {
  const { t } = useTranslation();
  return (
    <main id="main-content" className="flex min-h-screen flex-col items-center justify-center gap-4 px-6 text-center">
      <h1 className="page-title">404</h1>
      <p className="text-text-muted">{t('errors.not_found', 'This page does not exist.')}</p>
      <Link className="button-primary" to="/">{t('sidebar.dashboard', 'Dashboard')}</Link>
    </main>
  );
};

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
    else if (facet === 'tag') params.set('q', value);
    else params.set('q', value);
    navigate(`/graph?${params.toString()}`);
  };

  const { t } = useTranslation();

  return (
    <>
      <div className="min-h-screen bg-bg-main text-text-main dark:bg-dark-bg-main dark:text-dark-text-main font-sans selection:bg-action-primary/20">
        <a
          href="#main-content"
          className="fixed left-4 top-4 z-[100] -translate-y-24 rounded-lg bg-action-primary px-4 py-2 text-white transition-transform focus:translate-y-0"
        >
          {t('common.skip_to_content', 'Skip to content')}
        </a>
        <ErrorBoundary FallbackComponent={ErrorFallback}>
          <Suspense fallback={
            <div className="flex min-h-screen items-center justify-center text-sm text-text-muted">
              {t('common.loading', 'Loading…')}
            </div>
          }>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/graph" element={<GraphPage />} />
              <Route path="/data" element={<DataPage />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="*" element={<NotFound />} />
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
      <Router>
        <AppContent />
      </Router>
    </AdminAuthProvider>
  );
}

export default App;
