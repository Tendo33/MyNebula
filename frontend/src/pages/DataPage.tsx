import { useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Sidebar } from '../components/layout/Sidebar';
import { getGraphData } from '../api/graph';
import { GraphData } from '../types';
import { Loader2, ExternalLink } from 'lucide-react';

const DataPage = () => {
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

    return (
        <div className="flex min-h-screen bg-bg-main text-text-main">
            <Sidebar />

            <main className="flex-1 ml-60 flex flex-col min-w-0">
                <header className="flex items-center justify-between h-14 px-8 border-b border-border-light sticky top-0 bg-bg-main/95 backdrop-blur-sm z-40 transition-all">
                    <div className="flex items-center gap-3 select-none">
                        <h2 className="text-base font-semibold text-text-main tracking-tight">
                            {t('sidebar.data')}
                        </h2>
                        <div className="h-4 w-[1px] bg-border-light mx-1" />
                        <span className="text-sm text-text-muted">
                            {data?.total_nodes || 0} Repositories
                        </span>
                    </div>
                </header>

                <div className="flex-1 p-8 overflow-auto">
                    {loading ? (
                        <div className="flex items-center justify-center h-64">
                            <Loader2 className="animate-spin h-8 w-8 text-text-muted" />
                        </div>
                    ) : (
                        <div className="w-full overflow-hidden rounded-lg border border-border-light bg-white shadow-sm">
                            <div className="overflow-x-auto">
                                <table className="w-full text-left text-sm">
                                    <thead className="bg-bg-hover text-text-muted font-medium border-b border-border-light">
                                        <tr>
                                            <th className="px-4 py-3 whitespace-nowrap">Repository</th>
                                            <th className="px-4 py-3 whitespace-nowrap">Language</th>
                                            <th className="px-4 py-3 whitespace-nowrap text-right">Stars</th>
                                            <th className="px-4 py-3 whitespace-nowrap">Description</th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-border-light">
                                        {data?.nodes.map((repo) => (
                                            <tr key={repo.id} className="hover:bg-bg-hover/50 transition-colors">
                                                <td className="px-4 py-3 max-w-xs">
                                                     <div className="flex items-center gap-2">
                                                        <a href={repo.html_url} target="_blank" rel="noopener noreferrer" className="font-medium text-text-main hover:text-action-primary truncate block hover:underline">
                                                            {repo.full_name}
                                                        </a>
                                                        <ExternalLink className="w-3 h-3 text-text-muted flex-shrink-0" />
                                                     </div>
                                                </td>
                                                <td className="px-4 py-3 whitespace-nowrap">
                                                    {repo.language ? (
                                                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-gray-100 text-gray-800">
                                                            {repo.language}
                                                        </span>
                                                    ) : (
                                                        <span className="text-text-muted italic">Unknown</span>
                                                    )}
                                                </td>
                                                <td className="px-4 py-3 text-right font-mono text-text-dim">
                                                    {repo.stargazers_count.toLocaleString()}
                                                </td>
                                                <td className="px-4 py-3 max-w-md truncate text-text-muted">
                                                    {repo.description || <span className="italic text-text-dim">No description</span>}
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                         </div>
                    )}
                </div>
            </main>
        </div>
    );
};

export default DataPage;
