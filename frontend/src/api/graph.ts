import client from './client';
import { GraphData, TimelineData } from '../types';

export const getGraphData = async (params?: { include_edges?: boolean; min_similarity?: number }) => {
  const response = await client.get<GraphData>('/graph', { params });
  return response.data;
};

export const getTimelineData = async () => {
  const response = await client.get<TimelineData>('/graph/timeline');
  return response.data;
};
