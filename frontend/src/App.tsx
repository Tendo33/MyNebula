import { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Link, Routes, Route, useLocation, useNavigate } from 'react-router-dom';
import { ErrorBoundary } from 'react-error-boundary';
import { useTranslation } from 'react-i18next';
import { useTheme } from './hooks/useTheme';
import { GraphProvider, useGraph } from './contexts/GraphContext';
import { AdminAuthProvider } from './contexts/AdminAuthContext';
import { AppShell } from './components/layout/AppShell';
import { ErrorFallback } from './components/ui/ErrorFallback';
import CommandPalette from './components/ui/CommandPalette';
import { Button } from './components/ui/button';
import { TooltipProvider } from './components/ui/tooltip';
import useCommandPalette from './hooks/useCommandPalette';
import { ClusterInfo, GraphNode } from './types';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const GraphPage = lazy(() => import('./pages/GraphPage'));
const DataPage = lazy(() => import('./pages/DataPage'));
const Settings = lazy(() => import('./pages/Settings'));

const NotFound = () => {
  const { t } = useTranslation();
  return (
    <div className="flex flex-1 flex-col items-center justify-center gap-4 px-6 text-center">
      <h1 className="page-title">404</h1>
      <p className="text-muted-foreground">{t('errors.not_found')}</p>
      <Button nativeButton={false} render={<Link to="/" />}>
        {t('sidebar.dashboard')}
      </Button>
    </div>
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
  useTheme();

  return (
    <>
      <AppShell>
        <ErrorBoundary FallbackComponent={ErrorFallback}>
          <Suspense fallback={
            <div className="flex flex-1 items-center justify-center text-sm text-muted-foreground">
              {t('common.loading')}
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
      </AppShell>

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
      <TooltipProvider>
        <Router>
          <AppContent />
        </Router>
      </TooltipProvider>
    </AdminAuthProvider>
  );
}

export default App;
