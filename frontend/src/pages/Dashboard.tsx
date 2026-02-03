import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Sidebar } from '../components/layout/Sidebar';
import Graph3D from '../components/graph/Graph3D';
import Timeline from '../components/graph/Timeline';
import { SearchInput } from '../components/ui/SearchInput';

import { getGraphData } from '../api/graph';
import { startStarSync, getSyncStatus } from '../api/sync';
import { GraphData, GraphNode } from '../types';
import { RepoDetailsPanel } from '../components/graph/RepoDetailsPanel';


const Dashboard = () => {
  const { t } = useTranslation();
  const [data, setData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);


  useEffect(() => {
    // Determine if we should fetch real data (if API is available)
    // For now, we rely on Graph3D's mock data fallback if data is null
    // But let's try to fetch
    const loadData = async () => {
        try {
            setLoading(true);
            const graphData = await getGraphData();
            setData(graphData);
        } catch (error) {
            console.warn("Failed to fetch graph data, using mock", error);
            // Keep data null to trigger mock in Graph3D
        } finally {
            setLoading(false);
        }
    };
    loadData();
  }, []);

  const handleNodeClick = (node: GraphNode) => {
    setSelectedNode(node);
  };

  const handleCloseDetails = () => {
    setSelectedNode(null);
  };

  const handleSyncStars = async () => {
    try {
      setSyncing(true);
      const result = await startStarSync('incremental');
      console.log('Sync started:', result);

      // 轮询同步状态
      const pollStatus = async () => {
        try {
          const status = await getSyncStatus(result.task_id);
          console.log('Sync status:', status);

          if (status.status === 'completed') {
            // 同步完成，重新加载数据
            setSyncing(false);
            const graphData = await getGraphData();
            setData(graphData);
          } else if (status.status === 'failed') {
            setSyncing(false);
            console.error('Sync failed:', status.error_message);
            alert(status.error_message || '同步失败');
          } else {
            // 继续轮询
            setTimeout(pollStatus, 2000);
          }
        } catch (err) {
          console.error('Poll status error:', err);
          setSyncing(false);
        }
      };

      // 开始轮询
      setTimeout(pollStatus, 1000);

    } catch (error) {
      console.error('Failed to start sync:', error);
      setSyncing(false);
      alert('同步失败,请检查是否已配置 GITHUB_TOKEN');
    }
  };



  return (
    <div className="flex min-h-screen bg-nebula-bg text-nebula-text-main overflow-hidden">
        {/* Background Gradients */}
        <div className="fixed inset-0 pointer-events-none z-0">
             <div className="absolute -top-[20%] -left-[10%] w-[50%] h-[50%] bg-nebula-primary/5 blur-[120px] rounded-full" />
             <div className="absolute top-[40%] right-[0%] w-[40%] h-[40%] bg-nebula-secondary/5 blur-[120px] rounded-full" />
        </div>

      <Sidebar />

      <main className="flex-1 ml-72 p-4 h-screen flex flex-col gap-4 relative z-10">

        {/* Top Bar */}
        <header className="flex items-center justify-between p-4 bg-nebula-surface/50 backdrop-blur-md border border-nebula-border rounded-2xl shadow-lg">
           <div className="flex flex-col">
              <h2 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">
                  {t('dashboard.title')}
              </h2>
              <p className="text-xs text-nebula-text-muted">
                {data?.total_nodes
                  ? t('dashboard.subtitle', { count: data.total_nodes })
                  : t('dashboard.subtitle_infinite')}
              </p>
           </div>

           <div className="w-96">
                <SearchInput />
           </div>

           <div className="flex gap-2">
               <button
                 onClick={handleSyncStars}
                 disabled={syncing}
                 className={`px-4 py-2 rounded-xl text-sm font-medium transition-colors flex items-center gap-2 ${
                   syncing
                     ? 'bg-nebula-primary/5 text-nebula-text-muted cursor-not-allowed'
                     : 'bg-nebula-primary/10 text-nebula-primary hover:bg-nebula-primary/20'
                 }`}
               >
                   {syncing && (
                     <svg className="animate-spin h-4 w-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                       <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                       <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                     </svg>
                   )}
                   {syncing ? t('dashboard.syncing') || '同步中...' : t('dashboard.sync_button')}
               </button>
           </div>
        </header>

        {/* Visualization Area */}
        <section className="flex-1 min-h-0 relative rounded-2xl shadow-2xl shadow-black/50 border border-nebula-border overflow-hidden group">
            <Graph3D data={data} onNodeClick={handleNodeClick} />

            {/* Repo Details Panel */}
            {selectedNode && (
              <RepoDetailsPanel node={selectedNode} onClose={handleCloseDetails} />
            )}


            {/* Timeline Overlay */}
            <div className="absolute bottom-6 left-6 right-6 z-20 transition-transform duration-300 translate-y-4 group-hover:translate-y-0 opacity-80 group-hover:opacity-100">
                 <Timeline />
            </div>

            {/* Loading Indicator */}

            {loading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black/60 backdrop-blur-sm z-50">
                    <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-nebula-primary"></div>
                </div>
            )}
        </section>
      </main>
    </div>
  );
};

export default Dashboard;
