import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Sidebar } from '../components/layout/Sidebar';
import Graph3D from '../components/graph/Graph3D';
import Timeline from '../components/graph/Timeline';
import { SearchInput } from '../components/ui/SearchInput';
import { getGraphData } from '../api/graph';
import { startStarSync, getSyncStatus, startEmbedding, startClustering } from '../api/sync';
import { GraphData, GraphNode } from '../types';
import { RepoDetailsPanel } from '../components/graph/RepoDetailsPanel';
import { Loader2 } from 'lucide-react';

const Dashboard = () => {
  const { t } = useTranslation();
  const [data, setData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(false);
  const [syncing, setSyncing] = useState(false);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

  useEffect(() => {
    const loadData = async () => {
        try {
            setLoading(true);
            const graphData = await getGraphData();
            setData(graphData);
        } catch (error) {
            console.warn("Failed to fetch graph data, using mock", error);
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

  const [syncStep, setSyncStep] = useState<string>('');

  const waitForTaskComplete = async (taskId: number): Promise<boolean> => {
    return new Promise((resolve) => {
      const poll = async () => {
        try {
          const status = await getSyncStatus(taskId);
          if (status.status === 'completed') {
            resolve(true);
          } else if (status.status === 'failed') {
            console.error('Task failed:', status.error_message);
            resolve(false);
          } else {
            setTimeout(poll, 2000);
          }
        } catch (err) {
          console.error('Poll status error:', err);
          resolve(false);
        }
      };
      setTimeout(poll, 1000);
    });
  };

  const handleSyncStars = async () => {
    try {
      setSyncing(true);

      // 步骤1: 同步星标
      setSyncStep(t('dashboard.sync_step_stars'));
      const starsResult = await startStarSync('incremental');
      console.log('Stars sync started:', starsResult);

      const starsSuccess = await waitForTaskComplete(starsResult.task_id);
      if (!starsSuccess) {
        throw new Error('星标同步失败');
      }

      // 步骤2: 计算嵌入（增量，只处理新仓库）
      setSyncStep(t('dashboard.sync_step_embedding'));
      const embeddingResult = await startEmbedding();
      console.log('Embedding started:', embeddingResult);

      const embeddingSuccess = await waitForTaskComplete(embeddingResult.task_id);
      if (!embeddingSuccess) {
        throw new Error('嵌入计算失败');
      }

      // 步骤3: 运行聚类（生成3D坐标）
      setSyncStep(t('dashboard.sync_step_clustering'));
      const clusterResult = await startClustering();
      console.log('Clustering started:', clusterResult);

      const clusterSuccess = await waitForTaskComplete(clusterResult.task_id);
      if (!clusterSuccess) {
        throw new Error('聚类失败');
      }

      // 完成，刷新数据
      setSyncStep('');
      const graphData = await getGraphData();
      setData(graphData);

    } catch (error) {
      console.error('Sync failed:', error);
      alert(error instanceof Error ? error.message : '同步失败,请检查是否已配置 GITHUB_TOKEN');
    } finally {
      setSyncing(false);
      setSyncStep('');
    }
  };

  return (
    <div className="flex min-h-screen bg-bg-main text-text-main">
      <Sidebar />

      <main className="flex-1 ml-60 flex flex-col min-w-0">

        {/* Header / Top Bar for Page */}
        <header className="flex items-center justify-between h-14 px-8 border-b border-border-light sticky top-0 bg-bg-main/95 backdrop-blur-sm z-40 transition-all">
           <div className="flex items-center gap-3 select-none">
              <h2 className="text-base font-semibold text-text-main tracking-tight">
                  {t('dashboard.title')}
              </h2>
              <div className="h-4 w-[1px] bg-border-light mx-1" />
              <span className="text-sm text-text-muted">
                {data?.total_nodes
                  ? t('dashboard.subtitle', { count: data.total_nodes })
                  : t('dashboard.subtitle_infinite')}
              </span>
           </div>

           <div className="flex items-center gap-4">
                <div className="w-64 transition-all focus-within:w-72">
                    <SearchInput />
                </div>
                <button
                  onClick={handleSyncStars}
                  disabled={syncing}
                  className={`h-9 px-4 rounded-md text-sm font-medium transition-all flex items-center gap-2 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-black/20 hover:shadow-md active:scale-95 ${
                    syncing
                      ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                      : 'bg-black text-white hover:bg-gray-800 shadow-sm'
                  }`}
                >
                    {syncing && <Loader2 className="animate-spin h-4 w-4" />}
                    {syncing ? (syncStep || t('dashboard.syncing')) : t('dashboard.sync_button')}
                </button>
           </div>
        </header>

        {/* Content Area */}
        <section className="flex-1 relative flex flex-col">
            {/* Graph Container */}
            <div className="flex-1 relative bg-white">
                <Graph3D data={data} onNodeClick={handleNodeClick} />

                {selectedNode && (
                    <RepoDetailsPanel node={selectedNode} onClose={handleCloseDetails} />
                )}

                <div className="absolute bottom-6 left-6 right-6 z-20 pointer-events-none">
                     <div className="pointer-events-auto inline-block">
                        <Timeline />
                     </div>
                </div>

                {loading && (
                    <div className="absolute inset-0 flex items-center justify-center bg-white/50 z-50">
                        <Loader2 className="animate-spin h-8 w-8 text-action-primary" />
                    </div>
                )}
            </div>
        </section>
      </main>
    </div>
  );
};

export default Dashboard;
