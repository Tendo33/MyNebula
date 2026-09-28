import type { FC } from 'react';
import { useTranslation } from 'react-i18next';
import { Link, useNavigate } from 'react-router-dom';
import { ArrowRight, Layers } from 'lucide-react';

import { DashboardSkeleton } from '../components/ui/page-skeletons';
import { useDashboardQuery } from '../features/dashboard/hooks/useDashboardQuery';
import { EmptyState } from '../components/ui/EmptyState';
import { Button } from '../components/ui/button';
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '../components/ui/card';
import {
  Empty,
  EmptyContent,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from '../components/ui/empty';
import { ScrollArea } from '../components/ui/scroll-area';

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

const LanguageBar: FC<LanguageBarProps> = ({ language, count, percentage, color }) => {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-3">
        <span className="min-w-0 truncate text-sm text-foreground">{language}</span>
        <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
          {t('dashboard.repos_count', { count })}
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        <div
          className="h-full rounded-full"
          style={{ width: `${percentage}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
};

const ClusterRow: FC<ClusterRowProps> = ({ name, color, repoCount, keywords, onClick }) => {
  const { t } = useTranslation();
  return (
    <button
      type="button"
      onClick={onClick}
      className="grid w-full grid-cols-[auto_1fr_auto] items-baseline gap-x-3 gap-y-1 rounded-md px-2 py-3 text-left hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
    >
      <span
        className="mt-1.5 size-2 shrink-0 rounded-full"
        style={{ backgroundColor: color }}
        aria-hidden="true"
      />
      <span className="min-w-0 truncate text-sm font-semibold text-foreground">{name}</span>
      <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
        {t('dashboard.repos_count', { count: repoCount })}
      </span>
      {keywords.length > 0 ? (
        <span className="col-start-2 truncate text-xs text-muted-foreground">
          {keywords.slice(0, 4).join(' · ')}
        </span>
      ) : null}
    </button>
  );
};

const MetricCard = ({ label, value }: { label: string; value: number }) => (
  <Card>
    <CardHeader>
      <CardDescription>{label}</CardDescription>
      <CardTitle className="text-2xl tabular-nums">{value}</CardTitle>
    </CardHeader>
  </Card>
);

const Dashboard = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { stats, loading, error, retry } = useDashboardQuery();
  const hasCollection = (stats?.totalRepos ?? 0) > 0;
  const clusters = stats?.topClusters ?? [];
  const languages = stats?.topLanguages ?? [];

  return (
    <div className="min-h-0 flex-1 overflow-auto">
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
          <Empty>
            <EmptyHeader>
              <EmptyMedia variant="icon">
                <Layers />
              </EmptyMedia>
              <EmptyTitle>{t('dashboard.empty_title')}</EmptyTitle>
              <EmptyDescription>{t('dashboard.empty_hint')}</EmptyDescription>
            </EmptyHeader>
            <EmptyContent>
              <Button nativeButton={false} render={<Link to="/settings" />}>
                {t('common.sync_now')}
              </Button>
            </EmptyContent>
          </Empty>
        ) : (
          <div className="mx-auto flex max-w-6xl flex-col gap-4">
            <div className="grid grid-cols-2 gap-4 xl:grid-cols-4">
              <MetricCard label={t('dashboard.total_repos')} value={stats?.totalRepos ?? 0} />
              <MetricCard label={t('settings.synced')} value={stats?.embeddedRepos ?? 0} />
              <MetricCard label={t('dashboard.clusters')} value={stats?.totalClusters ?? 0} />
              <MetricCard label={t('common.languages')} value={languages.length} />
            </div>

            <div className="flex flex-wrap gap-2">
              <Button type="button" onClick={() => navigate('/graph')}>
                {t('dashboard.explore_graph')}
                <ArrowRight data-icon="inline-end" />
              </Button>
              <Button type="button" variant="outline" onClick={() => navigate('/data')}>
                {t('dashboard.browse_table')}
              </Button>
            </div>

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <Card className="min-h-0">
                <CardHeader>
                  <CardTitle>{t('dashboard.top_clusters')}</CardTitle>
                  <CardAction>
                    <Button type="button" variant="outline" size="sm" onClick={() => navigate('/graph')}>
                      {t('common.view_all')}
                      <ArrowRight data-icon="inline-end" />
                    </Button>
                  </CardAction>
                </CardHeader>
                <CardContent>
                  {clusters.length > 0 ? (
                    <ScrollArea className="h-80">
                      <ul className="flex flex-col pr-3">
                        {clusters.map((cluster) => (
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
                    </ScrollArea>
                  ) : (
                    <Empty className="border-0">
                      <EmptyHeader>
                        <EmptyTitle>{t('dashboard.no_data')}</EmptyTitle>
                      </EmptyHeader>
                    </Empty>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>{t('dashboard.language_distribution')}</CardTitle>
                </CardHeader>
                <CardContent>
                  {languages.length > 0 ? (
                    <ScrollArea className="h-80">
                      <div className="flex flex-col gap-4 pr-3">
                        {languages.map((lang) => (
                          <LanguageBar
                            key={lang.language}
                            language={lang.language}
                            count={lang.count}
                            percentage={lang.percentage}
                            color={lang.color}
                          />
                        ))}
                      </div>
                    </ScrollArea>
                  ) : (
                    <Empty className="border-0">
                      <EmptyHeader>
                        <EmptyTitle>{t('dashboard.no_data')}</EmptyTitle>
                      </EmptyHeader>
                    </Empty>
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </section>
    </div>
  );
};

export default Dashboard;
