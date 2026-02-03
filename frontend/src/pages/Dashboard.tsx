import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Sidebar } from '../components/layout/Sidebar';
import Graph3D from '../components/graph/Graph3D';
import Timeline from '../components/graph/Timeline';
import { SearchInput } from '../components/ui/SearchInput';

import { getGraphData } from '../api/graph';
import { GraphData } from '../types';

const Dashboard = () => {
  const { t } = useTranslation();
  const [data, setData] = useState<GraphData | null>(null);
  const [loading, setLoading] = useState(false);

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
               <button className="px-4 py-2 rounded-xl bg-nebula-primary/10 text-nebula-primary text-sm font-medium hover:bg-nebula-primary/20 transition-colors">
                   {t('dashboard.sync_button')}
               </button>
           </div>
        </header>

        {/* Visualization Area */}
        <section className="flex-1 min-h-0 relative rounded-2xl shadow-2xl shadow-black/50 border border-nebula-border overflow-hidden group">
            <Graph3D data={data} />

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
