import client from './client';
import { GraphData, TimelineData } from '../types';

export interface GetGraphDataParams {
  /** Whether to include similarity edges between nodes */
  include_edges?: boolean;
  /** Minimum similarity threshold for edges (0.5-0.95) */
  min_similarity?: number;
}

/**
 * Fetch graph data for visualization
 * @param params - Query parameters for graph data
 * @returns Complete graph data with nodes, edges, and clusters
 */
export const getGraphData = async (params?: GetGraphDataParams): Promise<GraphData> => {
  const response = await client.get<GraphData>('/graph', {
    params: {
      include_edges: params?.include_edges ?? false,
      min_similarity: params?.min_similarity ?? 0.7,
    },
  });
  return response.data;
};

/**
 * Fetch timeline data for visualization
 * @returns Timeline data grouped by month
 */
export const getTimelineData = async (): Promise<TimelineData> => {
  const response = await client.get<TimelineData>('/graph/timeline');
  return response.data;
};
