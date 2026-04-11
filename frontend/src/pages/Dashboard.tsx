import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { Sidebar } from '../components/layout/Sidebar';
import { LanguageSwitch } from '../components/layout/LanguageSwitch';
import { DashboardSkeleton } from '../components/ui/Skeleton';
import { Book, Code, Layers, TrendingUp, ArrowRight, Calendar, Hash, Tag } from 'lucide-react';
import { useDashboardQuery } from '../features/dashboard/hooks/useDashboardQuery';

// ============================================================================
// Types
// ============================================================================

interface StatCardProps {
  title: string;
  value: string | number;
  icon: React.ElementType;
  subValue?: string;
  trend?: { value: number; label: string };
  onClick?: () => void;
}

interface LanguageBarProps {
  language: string;
  count: number;
  percentage: number;
  color: string;
}

interface ClusterCardProps {
  name: string;
  color: string;
  repoCount: number;
  keywords: string[];
  onClick?: () => void;
}

// ============================================================================
// Sub Components
// ============================================================================

const StatCard: React.FC<StatCardProps> = ({ title, value, icon: Icon, subValue, trend, onClick }) => {
  const content = (
    <>
      <div>
        <p className="text-xs font-semibold uppercase tracking-[0.18em] text-text-dim">{title}</p>
        <h3 className="font-heading mt-3 text-3xl font-semibold text-text-main">{value}</h3>
        {subValue && <p className="mt-1.5 text-sm text-text-muted">{subValue}</p>}
        {trend && (
          <div className={`mt-3 flex items-center gap-1 text-xs font-medium ${trend.value >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            <TrendingUp className={`w-3 h-3 ${trend.value < 0 ? 'rotate-180' : ''}`} />
            <span>{trend.value >= 0 ? '+' : ''}{trend.value}% {trend.label}</span>
          </div>
        )}
      </div>
      <div className="rounded-2xl bg-bg-hover/85 p-3 text-text-main dark:bg-dark-bg-sidebar">
        <Icon className="w-5 h-5" />
      </div>
    </>
  );

  if (onClick) {
    return (
      <button
        type="button"
        onClick={onClick}
        className="panel-surface-strong flex items-start justify-between p-6 text-left transition-all hover:-translate-y-0.5"
      >
        {content}
      </button>
    );
  }

  return (
    <div className="panel-surface-strong flex items-start justify-between p-6 transition-all">
      {content}
    </div>
  );
};

const LanguageBar: React.FC<LanguageBarProps> = ({ language, count, percentage, color }) => {
  const { t } = useTranslation();
  return (
    <div className="group">
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-2">
          <div
            className="w-3 h-3 rounded-full ring-1 ring-black/10"
            style={{ backgroundColor: color }}
          />
          <span className="text-sm font-medium text-text-main dark:text-dark-text-main">{language}</span>
        </div>
        <span className="text-xs text-text-muted tabular-nums dark:text-dark-text-main/70">
          {t('dashboard.repos_count', { count })}
        </span>
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-bg-hover/90 dark:bg-dark-bg-sidebar">
        <div
          className="h-full rounded-full transition-all duration-500 group-hover:opacity-90"
          style={{
            width: `${percentage}%`,
            backgroundColor: color,
          }}
        />
      </div>
    </div>
);
};

const ClusterCard: React.FC<ClusterCardProps> = ({ name, color, repoCount, keywords, onClick }) => {
  const { t } = useTranslation();
  const content = (
    <div className="flex items-start gap-3">
      <div
        className="w-4 h-4 rounded-full mt-0.5 ring-1 ring-black/10 flex-shrink-0"
        style={{ backgroundColor: color }}
      />
      <div className="flex-1 min-w-0">
        <h4 className="font-heading text-sm font-semibold text-text-main truncate dark:text-dark-text-main">{name}</h4>
        <p className="text-xs text-text-muted mt-1 dark:text-dark-text-main/70">
          {t('common.repositories', { count: repoCount })}
        </p>
        {keywords.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-1.5">
            {keywords.slice(0, 4).map((keyword, idx) => (
              <span
                key={idx}
                className="rounded-full bg-bg-hover px-2 py-1 text-[10px] font-semibold text-text-muted dark:bg-dark-bg-sidebar dark:text-dark-text-main/70"
              >
                {keyword}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  if (onClick) {
    return (
      <button
        type="button"
        onClick={onClick}
        className="panel-surface p-5 text-left transition-all hover:-translate-y-px"
      >
        {content}
      </button>
    );
  }

  return (
    <div className="panel-surface p-4">
      {content}
    </div>
  );
};

// ============================================================================
// Main Component
// ============================================================================

const Dashboard = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { stats, activityData, maxActivity, loading, error, retry } = useDashboardQuery();

  // Navigate to graph with cluster filter
  const handleClusterClick = (clusterId: number) => {
    navigate(`/graph?cluster=${clusterId}`);
  };

  return (
    <div className="page-shell">
      <Sidebar />

      <main className="page-main">
        <header className="page-header">
          <div className="page-header-inner select-none">
            <div>
              <div className="section-kicker mb-1 px-0">{t('common.overview')}</div>
              <h2 className="page-title">
              {t('sidebar.dashboard')}
              </h2>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <LanguageSwitch />
            <button
              type="button"
              onClick={() => navigate('/graph')}
              className="header-action-ghost"
            >
              <span>{t('dashboard.explore_graph')}</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </header>

        <section className="page-content">
          {loading ? (
            <DashboardSkeleton />
          ) : error ? (
            <div className="max-w-6xl mx-auto min-h-[320px] flex flex-col items-center justify-center gap-3">
              <p className="text-sm text-red-600">{t('common.load_failed', 'Failed to load data')}</p>
              <button
                type="button"
                onClick={() => {
                  void retry();
                }}
                className="header-action"
              >
                {t('common.retry')}
              </button>
            </div>
          ) : (
            <div className="mx-auto max-w-[88rem] space-y-8">
              {/* Stats Grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <StatCard
                  title={t('dashboard.total_repos')}
                  value={stats?.totalRepos || 0}
                  icon={Book}
                  subValue={t('dashboard.starred_repos')}
                  onClick={() => navigate('/data')}
                />
                <StatCard
                  title={t('dashboard.total_topics')}
                  value={stats ? stats.totalTopics.toLocaleString() : '0'}
                  icon={Hash}
                  subValue={t('dashboard.unique_topics')}
                />
                <StatCard
                  title={t('dashboard.top_language')}
                  value={stats?.topLanguage || t('common.n_a')}
                  icon={Code}
                />
                <StatCard
                  title={t('dashboard.clusters')}
                  value={stats?.totalClusters || 0}
                  icon={Layers}
                  subValue={t('dashboard.knowledge_groups')}
                  onClick={() => navigate('/graph')}
                />
              </div>

              {/* Charts Row */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
                {/* Language Distribution */}
                <div className="panel-surface p-6">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="font-heading text-sm font-semibold text-text-main">
                      {t('dashboard.language_distribution')}
                    </h3>
                    <span className="text-xs text-text-muted">
                      {stats?.topLanguages?.length || 0} {t('common.languages')}
                    </span>
                  </div>

                  <div className="space-y-4">
                    {stats?.topLanguages?.map((lang, idx) => (
                      <LanguageBar
                        key={idx}
                        language={lang.language}
                        count={lang.count}
                        percentage={lang.percentage}
                        color={lang.color}
                      />
                    ))}
                  </div>

                  {(!stats?.topLanguages || stats.topLanguages.length === 0) && (
                    <div className="flex items-center justify-center h-32 text-text-muted text-sm">
                      {t('dashboard.no_data')}
                    </div>
                  )}
                </div>

                {/* Right Column: Activity + Topics */}
                <div className="flex flex-col gap-6">
                  {/* Activity Timeline */}
                  <div className="panel-surface p-6">
                    <div className="flex items-center justify-between mb-6">
                      <div className="flex items-center gap-2">
                        <Calendar className="w-4 h-4 text-text-muted" />
                        <h3 className="font-heading text-sm font-semibold text-text-main">
                          {t('dashboard.star_activity')}
                        </h3>
                      </div>
                      {stats?.recentActivity !== undefined && stats.recentActivity > 0 && (
                        <span className="text-xs text-green-600 bg-green-50 px-2 py-1 rounded-full font-medium">
                          +{stats.recentActivity} {t('dashboard.last_3_months')}
                        </span>
                      )}
                    </div>

                  {/* Activity bars */}
                  <div className="flex items-end gap-1.5 h-32 pt-8">
                    {activityData.map((data, idx) => {
                      const height = (data.count / maxActivity) * 100;
                      const isRecent = idx >= activityData.length - 3;
                      return (
                        <div
                          key={idx}
                          className="flex-1 flex flex-col items-center justify-end group h-full relative"
                        >
                          {/* Hover tooltip */}
                          <div className="absolute -top-7 left-1/2 -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity duration-200 pointer-events-none z-10">
                            <div className="bg-text-main text-bg-main text-[10px] px-2 py-1 rounded shadow-lg whitespace-nowrap">
                              <div className="font-medium">{t('dashboard.repos_count', { count: data.count })}</div>
                              <div className="text-bg-main/70">{data.date}</div>
                            </div>
                            <div className="w-2 h-2 bg-text-main rotate-45 absolute left-1/2 -translate-x-1/2 -bottom-1" />
                          </div>

                          {/* Bar */}
                          <button
                            type="button"
                            className={`w-full rounded-t transition-all duration-300 cursor-pointer hover:opacity-90 ${
                              isRecent
                                ? 'bg-gradient-to-t from-action-primary to-action-hover shadow-sm'
                                : 'bg-gradient-to-t from-action-primary/45 to-action-primary/20'
                            } focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/50`}
                            style={{ height: `${Math.max(height, 8)}%` }}
                            onClick={() => navigate(`/data?month=${data.date}`)}
                            aria-label={t('dashboard.repos_count', { count: data.count }) + ` (${data.date})`}
                          />
                        </div>
                      );
                    })}
                  </div>

                  {/* Date labels */}
                  {activityData.length > 0 && (
                    <div className="flex justify-between mt-3 text-[10px] text-text-muted font-medium">
                      <span>{activityData[0]?.date}</span>
                      <span className="text-text-dim">
                        {activityData[Math.floor(activityData.length / 2)]?.date}
                      </span>
                      <span>{activityData[activityData.length - 1]?.date}</span>
                    </div>
                  )}

                  {activityData.length === 0 && (
                    <div className="flex items-center justify-center h-32 text-text-muted text-sm">
                      {t('dashboard.no_activity')}
                    </div>
                  )}
                </div>

                {/* Popular Topics */}
                <div className="panel-surface p-6">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <Tag className="w-4 h-4 text-text-muted" />
                      <h3 className="font-heading text-sm font-semibold text-text-main">
                        {t('dashboard.popular_topics')}
                      </h3>
                    </div>
                    <span className="text-xs text-text-muted">
                      {t('dashboard.top_12')}
                    </span>
                  </div>

                  {/* Topics cloud */}
                  <div className="flex flex-wrap gap-2">
                    {stats?.topTopics?.map((item, idx) => {
                      // Color intensity based on count ranking
                      const totalTopics = stats?.topTopics?.length || 1;
                      const intensity = 1 - (idx / (totalTopics - 1 || 1)) * 0.6;
                      return (
                        <button
                          key={item.topic}
                          onClick={() => navigate(`/data?topic=${encodeURIComponent(item.topic)}`)}
                          className="group rounded-full px-3 py-1.5 text-sm font-semibold transition-all duration-200 hover:-translate-y-px hover:shadow-sm"
                          style={{
                            backgroundColor: `rgba(45, 89, 200, ${intensity * 0.11})`,
                            color: `rgba(35, 71, 163, ${0.76 + intensity * 0.2})`,
                            border: `1px solid rgba(45, 89, 200, ${intensity * 0.22})`,
                          }}
                        >
                          <span>{item.topic}</span>
                          <span className="ml-1.5 text-xs opacity-60">
                            {item.count}
                          </span>
                        </button>
                      );
                    })}
                  </div>

                  {(!stats?.topTopics || stats.topTopics.length === 0) && (
                    <div className="flex items-center justify-center h-20 text-text-muted text-sm">
                      {t('dashboard.no_topics')}
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Clusters Grid */}
            {stats?.topClusters && stats.topClusters.length > 0 && (
              <div>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-heading text-sm font-semibold text-text-main">
                    {t('dashboard.top_clusters')}
                  </h3>
                  <button
                    type="button"
                    onClick={() => navigate('/graph')}
                    className="header-action-ghost min-h-0 px-0 text-xs text-action-primary hover:bg-transparent hover:text-action-hover"
                  >
                    {t('common.view_all')}
                    <ArrowRight className="w-3 h-3" />
                  </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {stats.topClusters.map((cluster) => (
                    <ClusterCard
                      key={cluster.id}
                      name={cluster.name || `Cluster ${cluster.id}`}
                      color={cluster.color || '#6B7280'}
                      repoCount={cluster.repo_count}
                      keywords={cluster.keywords || []}
                      onClick={() => handleClusterClick(cluster.id)}
                    />
                  ))}
                </div>
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
