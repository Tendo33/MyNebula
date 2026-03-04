import client from '../client';
import { GraphData, GraphEdge, TimelineData } from '../../types';

export interface GraphEdgesPage {
  edges: GraphEdge[];
  next_cursor: number | null;
  version: string;
  generated_at?: string;
  request_id?: string;
}

export const getGraphDataV2 = async (params?: {
  version?: string;
  include_edges?: boolean;
}): Promise<GraphData> => {
  const response = await client.get<GraphData>('/v2/graph', {
    params: {
      version: params?.version ?? 'active',
      include_edges: params?.include_edges ?? false,
    },
  });
  return response.data;
};

export const getTimelineDataV2 = async (
  version: string = 'active'
): Promise<TimelineData> => {
  const response = await client.get<TimelineData>('/v2/graph/timeline', { params: { version } });
  return response.data;
};

export const getGraphEdgesPageV2 = async (
  params?: { version?: string; cursor?: number; limit?: number }
): Promise<GraphEdgesPage> => {
  const response = await client.get<GraphEdgesPage>('/v2/graph/edges', {
    params: {
      version: params?.version ?? 'active',
      cursor: params?.cursor ?? 0,
      limit: params?.limit ?? 1000,
    },
  });
  return response.data;
};

export const rebuildGraphV2 = async (): Promise<GraphData> => {
  const response = await client.post<GraphData>('/v2/graph/rebuild');
  return response.data;
};
