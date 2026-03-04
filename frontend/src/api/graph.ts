import { GraphData, GraphEdge, TimelineData } from '../types';
import {
  getGraphDataV2,
  getGraphEdgesPageV2,
  getTimelineDataV2,
  rebuildGraphV2,
} from './v2/graph';

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
  return getGraphDataV2({ version: 'active', include_edges: false });
};

/**
 * Fetch timeline data for visualization
 * @returns Timeline data grouped by month
 */
export const getTimelineData = async (): Promise<TimelineData> => {
  return getTimelineDataV2('active');
};

/**
 * Fetch graph edges separately so nodes can render first.
 */
export const getGraphEdges = async (
  params?: GetGraphEdgesParams
): Promise<GraphEdge[]> => {
  const limit = Math.max(100, Math.min(5000, params?.max_nodes ?? 1000));
  const edges: GraphEdge[] = [];
  let cursor: number | null = 0;

  while (cursor != null) {
    const page = await getGraphEdgesPageV2({
      version: 'active',
      cursor,
      limit,
    });
    edges.push(...page.edges);
    cursor = page.next_cursor;
  }

  return edges;
};

export const rebuildGraph = async (): Promise<GraphData> => {
  return rebuildGraphV2();
};
