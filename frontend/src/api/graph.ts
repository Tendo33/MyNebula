import client from './client';
import { GraphData, TimelineData } from '../types';

export const getGraphData = async (params?: { include_edges?: boolean; min_similarity?: number }) => {
  const token = localStorage.getItem("token");
	const response = await client.get<GraphData>("/graph", {
		params: { ...params, token },
	});
  return response.data;
};

export const getTimelineData = async () => {
  const token = localStorage.getItem("token");
	const response = await client.get<TimelineData>("/graph/timeline", {
		params: { token },
	});
  return response.data;
};
