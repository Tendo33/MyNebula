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
        <p className="text-sm font-medium text-text-muted">{title}</p>
        <h3 className="text-2xl font-bold text-text-main mt-2">{value}</h3>
        {subValue && <p className="text-xs text-text-dim mt-1">{subValue}</p>}
        {trend && (
          <div className={`flex items-center gap-1 mt-2 text-xs ${trend.value >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            <TrendingUp className={`w-3 h-3 ${trend.value < 0 ? 'rotate-180' : ''}`} />
            <span>{trend.value >= 0 ? '+' : ''}{trend.value}% {trend.label}</span>
          </div>
        )}
      </div>
      <div className="p-2 bg-bg-hover rounded-md text-text-main dark:bg-dark-bg-sidebar">
        <Icon className="w-5 h-5" />
      </div>
    </>
  );

  if (onClick) {
    return (
      <button
        type="button"
        onClick={onClick}
        className="bg-bg-main/90 p-6 rounded-xl border border-border-light shadow-sm flex items-start justify-between transition-all backdrop-blur-sm cursor-pointer hover:shadow-md hover:border-border-light hover:-translate-y-0.5 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30 dark:bg-dark-bg-main/90 dark:border-dark-border"
      >
        {content}
      </button>
    );
  }

  return (
    <div className="bg-bg-main/90 p-6 rounded-xl border border-border-light shadow-sm flex items-start justify-between transition-all backdrop-blur-sm dark:bg-dark-bg-main/90 dark:border-dark-border">
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
      <div className="h-2 bg-bg-hover rounded-full overflow-hidden dark:bg-dark-bg-sidebar">
        <div
          className="h-full rounded-full transition-all duration-500 group-hover:opacity-80"
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
        <h4 className="text-sm font-semibold text-text-main truncate dark:text-dark-text-main">{name}</h4>
        <p className="text-xs text-text-muted mt-0.5 dark:text-dark-text-main/70">
          {t('common.repositories', { count: repoCount })}
        </p>
        {keywords.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-2">
            {keywords.slice(0, 4).map((keyword, idx) => (
              <span
                key={idx}
                className="text-[10px] px-1.5 py-0.5 bg-bg-hover text-text-muted rounded-full dark:bg-dark-bg-sidebar dark:text-dark-text-main/70"
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
        className="p-4 rounded-xl border border-border-light bg-bg-main/90 hover:shadow-md hover:border-border-light hover:-translate-y-0.5 transition-all cursor-pointer backdrop-blur-sm text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30 dark:bg-dark-bg-main/90 dark:border-dark-border"
      >
        {content}
      </button>
    );
  }

  return (
    <div className="p-4 rounded-xl border border-border-light bg-bg-main/90 backdrop-blur-sm dark:bg-dark-bg-main/90 dark:border-dark-border">
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
    <div className="flex min-h-screen bg-bg-main text-text-main dark:bg-dark-bg-main dark:text-dark-text-main">
      <Sidebar />

      <main className="flex-1 flex flex-col min-w-0" style={{ marginLeft: 'var(--sidebar-width, 240px)' }}>
        <header className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between min-h-[3.5rem] px-4 sm:px-8 py-3 sm:py-0 border-b border-border-light sticky top-0 bg-bg-main/95 backdrop-blur-sm z-40 dark:bg-dark-bg-main/95 dark:border-dark-border">
          <div className="flex items-center gap-3 select-none">
            <h2 className="text-base font-semibold text-text-main tracking-tight">
              {t('sidebar.dashboard')}
            </h2>
          </div>

          <div className="flex items-center gap-3">
            <LanguageSwitch />
            <button
              onClick={() => navigate('/graph')}
              className="flex items-center gap-2 text-sm text-text-muted hover:text-text-main transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30 dark:text-dark-text-main/70 dark:hover:text-dark-text-main"
            >
              <span>{t('dashboard.explore_graph')}</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </header>

        <section className="flex-1 p-4 sm:p-8 overflow-auto">
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
                className="px-3 py-1.5 rounded border border-border-light text-sm hover:bg-bg-hover dark:border-dark-border"
              >
                {t('common.retry')}
              </button>
            </div>
          ) : (
            <div className="max-w-6xl mx-auto space-y-8">
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
                <div className="bg-bg-main/90 p-6 rounded-xl border border-border-light shadow-sm backdrop-blur-sm dark:bg-dark-bg-main/90 dark:border-dark-border">
                  <div className="flex items-center justify-between mb-6">
                  <h3 className="text-sm font-semibold text-text-main">
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
                  <div className="bg-bg-main/90 p-6 rounded-xl border border-border-light shadow-sm backdrop-blur-sm dark:bg-dark-bg-main/90 dark:border-dark-border">
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-2">
                      <Calendar className="w-4 h-4 text-text-muted" />
                      <h3 className="text-sm font-semibold text-text-main">
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
                          <div
                            className={`w-full rounded-t transition-all duration-300 cursor-pointer hover:opacity-90 ${
                              isRecent
                                ? 'bg-gradient-to-t from-action-primary to-action-hover shadow-sm'
                                : 'bg-gradient-to-t from-action-primary/50 to-action-primary/30'
                            }`}
                            style={{ height: `${Math.max(height, 8)}%` }}
                            onClick={() => navigate(`/data?month=${data.date}`)}
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
                <div className="bg-bg-main/90 p-6 rounded-xl border border-border-light shadow-sm backdrop-blur-sm dark:bg-dark-bg-main/90 dark:border-dark-border">
                  <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-2">
                      <Tag className="w-4 h-4 text-text-muted" />
                      <h3 className="text-sm font-semibold text-text-main">
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
                          className="group px-3 py-1.5 rounded-full text-sm font-medium transition-all duration-200 hover:shadow-md hover:scale-105"
                          style={{
                            backgroundColor: `rgba(59, 130, 246, ${intensity * 0.15})`,
                            color: `rgba(37, 99, 235, ${0.7 + intensity * 0.3})`,
                            border: `1px solid rgba(59, 130, 246, ${intensity * 0.3})`,
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
                  <h3 className="text-sm font-semibold text-text-main">
                    {t('dashboard.top_clusters')}
                  </h3>
                  <button
                    onClick={() => navigate('/graph')}
                    className="text-xs text-action-primary hover:text-action-hover transition-colors flex items-center gap-1"
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
