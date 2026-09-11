import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { Sidebar } from '../components/layout/Sidebar';
import { LanguageSwitch } from '../components/layout/LanguageSwitch';
import { DashboardSkeleton } from '../components/ui/Skeleton';
import { ArrowRight } from 'lucide-react';
import { useDashboardQuery } from '../features/dashboard/hooks/useDashboardQuery';
import { EmptyState } from '../components/ui/EmptyState';

interface LanguageBarProps {
  language: string;
  count: number;
  percentage: number;
  color: string;
}

interface ClusterRowProps {
  name: string;
  color: string;
  repoCount: number;
  keywords: string[];
  onClick?: () => void;
}

const LanguageBar: React.FC<LanguageBarProps> = ({ language, count, percentage, color }) => {
  const { t } = useTranslation();
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-3">
        <span className="min-w-0 truncate text-sm text-text-main">{language}</span>
        <span className="shrink-0 text-xs tabular-nums text-text-muted">
          {t('dashboard.repos_count', { count })}
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-bg-hover">
        <div
          className="h-full rounded-full"
          style={{ width: `${percentage}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
};

const ClusterRow: React.FC<ClusterRowProps> = ({ name, color, repoCount, keywords, onClick }) => {
  const { t } = useTranslation();
  return (
    <button
      type="button"
      onClick={onClick}
      className="grid w-full grid-cols-[auto_1fr_auto] items-baseline gap-x-3 gap-y-1 rounded-xl px-2 py-3 text-left transition-colors hover:bg-bg-hover/80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary motion-reduce:transition-none"
    >
      <span
        className="mt-1.5 h-2 w-2 shrink-0 rounded-full"
        style={{ backgroundColor: color }}
        aria-hidden="true"
      />
      <span className="min-w-0 truncate font-heading text-sm font-semibold text-text-main">
        {name}
      </span>
      <span className="shrink-0 text-xs tabular-nums text-text-muted">
        {t('dashboard.repos_count', { count: repoCount })}
      </span>
      {keywords.length > 0 ? (
        <span className="col-start-2 truncate text-xs text-text-muted">
          {keywords.slice(0, 4).join(' · ')}
        </span>
      ) : null}
    </button>
  );
};

const Dashboard = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { stats, activityData, maxActivity, loading, error, retry } = useDashboardQuery();
  const hasCollection = (stats?.totalRepos ?? 0) > 0;

  return (
    <div className="page-shell">
      <Sidebar />

      <main id="main-content" className="page-main">
        <header className="page-header">
          <h1 className="page-title">{t('sidebar.dashboard')}</h1>
          <LanguageSwitch />
        </header>

        <section className="page-content">
          {loading ? (
            <DashboardSkeleton />
          ) : error ? (
            <EmptyState
              title={t('common.load_failed_dashboard')}
              actionType="button"
              actionLabel={t('common.retry')}
              onAction={() => {
                void retry();
              }}
            />
          ) : !hasCollection ? (
            <EmptyState
              title={t('dashboard.empty_title')}
              description={t('dashboard.empty_hint')}
              actionTo="/settings"
              actionLabel={t('common.sync_now')}
            />
          ) : (
            <div className="mx-auto max-w-5xl">
              <div className="border-b border-border-light pb-8">
                <h2 className="font-heading text-[32px] font-semibold leading-10 tracking-[-1.28px] text-text-main">
                  {t('dashboard.collection_heading', { count: stats?.totalRepos ?? 0 })}
                </h2>
                <p className="mt-3 max-w-prose text-sm text-text-muted">
                  {t('dashboard.collection_meta', {
                    clusters: stats?.totalClusters ?? 0,
                    topics: stats?.totalTopics ?? 0,
                  })}
                  {stats?.topLanguage ? ` · ${stats.topLanguage}` : ''}
                </p>
                <div className="mt-6 flex flex-wrap gap-2">
                  <button type="button" onClick={() => navigate('/graph')} className="header-action">
                    {t('dashboard.explore_graph')}
                    <ArrowRight className="h-4 w-4" />
                  </button>
                  <button
                    type="button"
                    onClick={() => navigate('/data')}
                    className="header-action-ghost"
                  >
                    {t('dashboard.browse_table')}
                  </button>
                </div>
              </div>

              <div className="mt-10 grid grid-cols-1 gap-12 lg:grid-cols-5">
                <div className="lg:col-span-2">
                  <h3 className="font-heading text-sm font-semibold text-text-main">
                    {t('dashboard.language_distribution')}
                  </h3>
                  <div className="mt-5 flex flex-col gap-4">
                    {stats?.topLanguages?.map((lang) => (
                      <LanguageBar
                        key={lang.language}
                        language={lang.language}
                        count={lang.count}
                        percentage={lang.percentage}
                        color={lang.color}
                      />
                    ))}
                    {(!stats?.topLanguages || stats.topLanguages.length === 0) && (
                      <p className="text-sm text-text-muted">{t('dashboard.no_data')}</p>
                    )}
                  </div>
                </div>

                <div className="flex flex-col gap-12 lg:col-span-3">
                  <div>
                    <div className="flex items-baseline justify-between gap-3">
                      <h3 className="font-heading text-sm font-semibold text-text-main">
                        {t('dashboard.star_activity')}
                      </h3>
                      {stats?.recentActivity !== undefined && stats.recentActivity > 0 && (
                        <span className="text-xs tabular-nums text-text-muted">
                          +{stats.recentActivity} {t('dashboard.last_3_months')}
                        </span>
                      )}
                    </div>
                    {activityData.length > 0 ? (
                      <>
                        <div className="mt-6 flex h-28 items-end gap-1">
                          {activityData.map((data, idx) => {
                            const height = (data.count / maxActivity) * 100;
                            const isRecent = idx >= activityData.length - 3;
                            return (
                              <button
                                key={data.date}
                                type="button"
                                className={`min-h-1 flex-1 rounded-sm transition-opacity hover:opacity-80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary ${
                                  isRecent ? 'bg-action-primary' : 'bg-action-primary/40'
                                }`}
                                style={{ height: `${Math.max(height, 6)}%` }}
                                onClick={() => navigate(`/data?month=${data.date}`)}
                                aria-label={`${t('dashboard.repos_count', { count: data.count })} (${data.date})`}
                              />
                            );
                          })}
                        </div>
                        <div className="mt-2 flex justify-between text-xs text-text-muted">
                          <span>{activityData[0]?.date}</span>
                          <span>{activityData[activityData.length - 1]?.date}</span>
                        </div>
                      </>
                    ) : (
                      <p className="mt-5 text-sm text-text-muted">{t('dashboard.no_activity')}</p>
                    )}
                  </div>

                  <div>
                    <h3 className="font-heading text-sm font-semibold text-text-main">
                      {t('dashboard.popular_topics')}
                    </h3>
                    {stats?.topTopics && stats.topTopics.length > 0 ? (
                      <div className="mt-4 flex flex-wrap gap-2">
                        {stats.topTopics.map((item) => (
                          <button
                            key={item.topic}
                            type="button"
                            onClick={() =>
                              navigate(`/data?topic=${encodeURIComponent(item.topic)}`)
                            }
                            className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border border-border-light bg-bg-sidebar px-3 py-1.5 text-sm text-text-main transition-colors hover:border-action-primary hover:bg-bg-hover focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary"
                          >
                            {item.topic}
                            <span className="tabular-nums text-text-muted">{item.count}</span>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <p className="mt-4 text-sm text-text-muted">{t('dashboard.no_topics')}</p>
                    )}
                  </div>
                </div>
              </div>

              {stats?.topClusters && stats.topClusters.length > 0 && (
                <div className="mt-12 border-t border-border-light pt-10">
                  <div className="flex items-baseline justify-between gap-3">
                    <h3 className="font-heading text-sm font-semibold text-text-main">
                      {t('dashboard.top_clusters')}
                    </h3>
                    <button
                      type="button"
                      onClick={() => navigate('/graph')}
                      className="header-action-ghost"
                    >
                      {t('common.view_all')}
                      <ArrowRight className="h-3.5 w-3.5" />
                    </button>
                  </div>
                  <ul className="mt-2 divide-y divide-border-light">
                    {stats.topClusters.map((cluster) => (
                      <li key={cluster.id}>
                        <ClusterRow
                          name={cluster.name || `Cluster ${cluster.id}`}
                          color={cluster.color || '#8f8f8f'}
                          repoCount={cluster.repo_count}
                          keywords={cluster.keywords || []}
                          onClick={() => navigate(`/graph?cluster=${cluster.id}`)}
                        />
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </section>
      </main>
    </div>
  );
};

export default Dashboard;
