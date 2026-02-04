import { useEffect, useState, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { Sidebar } from '../components/layout/Sidebar';
import { getGraphData } from '../api/graph';
import { GraphData } from '../types';
import { Loader2, Book, Star, Code, Layers } from 'lucide-react';

const StatCard = ({ title, value, icon: Icon, subValue }: { title: string, value: string | number, icon: any, subValue?: string }) => (
  <div className="bg-white p-6 rounded-lg border border-border-light shadow-sm flex items-start justify-between">
    <div>
      <p className="text-sm font-medium text-text-muted">{title}</p>
      <h3 className="text-2xl font-bold text-text-main mt-2">{value}</h3>
      {subValue && <p className="text-xs text-text-dim mt-1">{subValue}</p>}
    </div>
    <div className="p-2 bg-bg-hover rounded-md text-text-main">
      <Icon className="w-5 h-5" />
    </div>
  </div>
);

const Dashboard = () => {
  const { t } = useTranslation();
  const [data, setData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadData = async () => {
        try {
            setLoading(true);
            const graphData = await getGraphData();
            setData(graphData);
        } catch (error) {
            console.warn("Failed to fetch graph data", error);
        } finally {
            setLoading(false);
        }
    };
    loadData();
  }, []);

  const stats = useMemo(() => {
    if (!data) return null;

    const totalRepos = data.total_nodes;
    const totalStars = data.nodes.reduce((acc, node) => acc + node.stargazers_count, 0);
    const totalClusters = data.total_clusters;

    const languages: Record<string, number> = {};
    data.nodes.forEach(node => {
        if (node.language) {
            languages[node.language] = (languages[node.language] || 0) + 1;
        }
    });
    const topLanguage = Object.entries(languages).sort((a, b) => b[1] - a[1])[0];

    return {
        totalRepos,
        totalStars,
        totalClusters,
        topLanguage: topLanguage ? `${topLanguage[0]} (${topLanguage[1]})` : 'N/A'
    };
  }, [data]);

  return (
    <div className="flex min-h-screen bg-bg-main text-text-main">
      <Sidebar />

      <main className="flex-1 ml-60 flex flex-col min-w-0">
        <header className="flex items-center justify-between h-14 px-8 border-b border-border-light sticky top-0 bg-bg-main/95 backdrop-blur-sm z-40">
           <div className="flex items-center gap-3 select-none">
              <h2 className="text-base font-semibold text-text-main tracking-tight">
                  {t('sidebar.dashboard')}
              </h2>
           </div>
        </header>

        <section className="flex-1 p-8 overflow-auto">
             {loading ? (
                <div className="flex items-center justify-center h-full min-h-[400px]">
                    <Loader2 className="animate-spin h-8 w-8 text-action-primary" />
                </div>
             ) : (
                 <div className="max-w-5xl mx-auto space-y-8">
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                        <StatCard
                            title={t('sidebar.data')}
                            value={stats?.totalRepos || 0}
                            icon={Book}
                            subValue="Total Repositories"
                        />
                        <StatCard
                            title="Total Stars"
                            value={stats?.totalStars.toLocaleString() || 0}
                            icon={Star}
                        />
                        <StatCard
                            title="Top Language"
                            value={stats?.topLanguage || '-'}
                            icon={Code}
                        />
                        <StatCard
                            title="Clusters"
                            value={stats?.totalClusters || 0}
                            icon={Layers}
                            subValue="Knowledge Groups"
                        />
                    </div>

                    {/* Placeholder for more widgets */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div className="bg-white p-6 rounded-lg border border-border-light shadow-sm min-h-[300px] flex items-center justify-center text-text-muted">
                            Stats Visualization Placeholder
                        </div>
                         <div className="bg-white p-6 rounded-lg border border-border-light shadow-sm min-h-[300px] flex items-center justify-center text-text-muted">
                            Recent Activity Placeholder
                        </div>
                    </div>
                 </div>
             )}
        </section>
      </main>
    </div>
  );
};

export default Dashboard;
