import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';

import { getDashboardV2 } from '../../../api/v2/dashboard';
import { getTimelineDataV2 } from '../../../api/v2/graph';
import { queryKeys } from '../../../lib/queryKeys';

const getLanguageColor = (language: string): string => {
  const colors: Record<string, string> = {
    JavaScript: '#f1e05a',
    TypeScript: '#3178c6',
    Python: '#3572A5',
    Java: '#b07219',
    Go: '#00ADD8',
    Rust: '#dea584',
    Ruby: '#701516',
    PHP: '#4F5D95',
    'C++': '#f34b7d',
    C: '#555555',
    'C#': '#178600',
    Swift: '#F05138',
    Kotlin: '#A97BFF',
    Scala: '#c22d40',
    Shell: '#89e051',
    Vue: '#41b883',
    HTML: '#e34c26',
    CSS: '#563d7c',
    Dart: '#00B4AB',
    Lua: '#000080',
  };
  return colors[language] || '#6B7280';
};

export const useDashboardQuery = () => {
  const dashboardQuery = useQuery({
    queryKey: ['v2-dashboard'],
    queryFn: getDashboardV2,
    staleTime: 20_000,
  });
  const timelineQuery = useQuery({
    queryKey: [...queryKeys.timeline(), 'dashboard'],
    queryFn: () => getTimelineDataV2('active'),
    staleTime: 20_000,
  });

  const stats = useMemo(() => {
    const dashboard = dashboardQuery.data;
    const timeline = timelineQuery.data;
    if (!dashboard) return null;

    const topLanguages = dashboard.top_languages.map(({ language, count }) => ({
        language,
        count,
        percentage: dashboard.summary.total_repos > 0 ? (count / dashboard.summary.total_repos) * 100 : 0,
        color: getLanguageColor(language),
      }));

    const recentMonths = timeline?.points.slice(-3) ?? [];
    const recentActivity = recentMonths.reduce((sum, point) => sum + point.count, 0);

    return {
      totalRepos: dashboard.summary.total_repos,
      totalTopics: dashboard.summary.total_topics,
      totalClusters: dashboard.summary.total_clusters,
      totalEdges: dashboard.summary.total_edges,
      topLanguages,
      topLanguage:
        topLanguages[0] != null
          ? `${topLanguages[0].language} (${topLanguages[0].count})`
          : null,
      topClusters: dashboard.top_clusters,
      topTopics: dashboard.top_topics,
      recentActivity,
    };
  }, [dashboardQuery.data, timelineQuery.data]);

  const activityData = useMemo(() => {
    const points = timelineQuery.data?.points ?? [];
    return points.slice(-12).map((point) => ({
      date: point.date,
      count: point.count,
      languages: point.top_languages,
    }));
  }, [timelineQuery.data]);

  const maxActivity = useMemo(
    () => Math.max(...activityData.map((item) => item.count), 1),
    [activityData]
  );

  return {
    stats,
    activityData,
    maxActivity,
    timelineData: timelineQuery.data,
    loading: dashboardQuery.isLoading || timelineQuery.isLoading,
    error:
      dashboardQuery.error ??
      timelineQuery.error ??
      null,
    retry: async () => {
      await Promise.all([
        dashboardQuery.refetch(),
        timelineQuery.refetch(),
      ]);
    },
  };
};
