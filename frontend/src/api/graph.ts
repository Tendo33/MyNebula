import client from './client';
import { GraphData, GraphEdge, TimelineData } from '../types';

export interface GetGraphEdgesParams {
  min_similarity?: number;
  k?: number;
  max_nodes?: number;
  adaptive?: boolean;
}

/**
 * Fetch graph data for visualization
 * @returns Complete graph data with nodes, edges, and clusters
 */
export const getGraphData = async (): Promise<GraphData> => {
  const response = await client.get<GraphData>('/graph');
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

/**
 * Fetch graph edges separately so nodes can render first.
 */
export const getGraphEdges = async (
  params?: GetGraphEdgesParams
): Promise<GraphEdge[]> => {
  const response = await client.get<GraphEdge[]>('/graph/edges', {
    params: {
      min_similarity: params?.min_similarity,
      k: params?.k ?? 8,
      max_nodes: params?.max_nodes ?? 1000,
      adaptive: params?.adaptive ?? true,
    },
  });
  return response.data;
};
