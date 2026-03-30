export const queryKeys = {
  graphData: (version = 'active') => ['v2', 'graph', version] as const,
  graphEdges: (version: string, nonce: number) =>
    ['v2', 'graph-edges', version, nonce] as const,
  timeline: (version = 'active') => ['v2', 'timeline', version] as const,
  dashboard: () => ['v2', 'dashboard'] as const,
  dataRepos: (...params: unknown[]) => ['v2', 'data-repos', ...params] as const,
};
