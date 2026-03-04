import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';

import { getDashboardV2 } from '../../../api/v2/dashboard';
import { getGraphDataV2, getTimelineDataV2 } from '../../../api/v2/graph';

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
  const graphQuery = useQuery({
    queryKey: ['v2-dashboard-graph'],
    queryFn: () => getGraphDataV2({ version: 'active', include_edges: false }),
    staleTime: 20_000,
  });
  const timelineQuery = useQuery({
    queryKey: ['v2-dashboard-timeline'],
    queryFn: () => getTimelineDataV2('active'),
    staleTime: 20_000,
  });

  const stats = useMemo(() => {
    const dashboard = dashboardQuery.data;
    const graph = graphQuery.data;
    const timeline = timelineQuery.data;
    if (!dashboard || !graph) return null;

    const topicCounts: Record<string, number> = {};
    const languageCounts: Record<string, number> = {};

    graph.nodes.forEach((node) => {
      if (node.language) {
        languageCounts[node.language] = (languageCounts[node.language] || 0) + 1;
      }
      if (Array.isArray(node.topics)) {
        node.topics.forEach((topic) => {
          const normalizedTopic = topic.toLowerCase();
          topicCounts[normalizedTopic] = (topicCounts[normalizedTopic] || 0) + 1;
        });
      }
    });

    const topLanguages = Object.entries(languageCounts)
      .sort((left, right) => right[1] - left[1])
      .slice(0, 8)
      .map(([language, count]) => ({
        language,
        count,
        percentage: graph.total_nodes > 0 ? (count / graph.total_nodes) * 100 : 0,
        color: getLanguageColor(language),
      }));

    const topClusters = dashboard.top_clusters.map((cluster) => {
      const fullCluster = graph.clusters.find((item) => item.id === cluster.id);
      return {
        ...cluster,
        keywords: fullCluster?.keywords ?? [],
      };
    });

    const recentMonths = timeline?.points.slice(-3) ?? [];
    const recentActivity = recentMonths.reduce((sum, point) => sum + point.count, 0);

    const topTopics = Object.entries(topicCounts)
      .sort((left, right) => right[1] - left[1])
      .slice(0, 12)
      .map(([topic, count]) => ({ topic, count }));

    return {
      totalRepos: dashboard.summary.total_repos,
      totalTopics: Object.keys(topicCounts).length,
      totalClusters: dashboard.summary.total_clusters,
      totalEdges: dashboard.summary.total_edges,
      topLanguages,
      topLanguage:
        topLanguages[0] != null
          ? `${topLanguages[0].language} (${topLanguages[0].count})`
          : null,
      topClusters,
      topTopics,
      recentActivity,
    };
  }, [dashboardQuery.data, graphQuery.data, timelineQuery.data]);

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
    rawData: graphQuery.data,
    timelineData: timelineQuery.data,
    loading: dashboardQuery.isLoading || graphQuery.isLoading || timelineQuery.isLoading,
    error:
      dashboardQuery.error ??
      graphQuery.error ??
      timelineQuery.error ??
      null,
  };
};
