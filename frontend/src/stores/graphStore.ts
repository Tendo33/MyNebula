import { create } from 'zustand';
import { GraphNode } from '../types';

interface GraphFiltersState {
  selectedClusters: Set<number>;
  selectedStarLists: Set<number>;
  searchQuery: string;
  timeRange: [number, number] | null;
  minStars: number;
  languages: Set<string>;
}

export interface GraphSettingsState {
  showTrajectories: boolean;
  hqRendering: boolean;
  maxClusters: number;
  minClusters: number;
  relatedMinSemantic: number;
}

interface GraphUiState {
  selectedNode: GraphNode | null;
  filters: GraphFiltersState;
  settings: GraphSettingsState;
  syncing: boolean;
  syncStep: string;
  setSelectedNode: (node: GraphNode | null) => void;
  updateSettings: (settings: Partial<GraphSettingsState>) => void;
  setSearchQuery: (query: string) => void;
  setTimeRange: (range: [number, number] | null) => void;
  setMinStars: (min: number) => void;
  setSelectedLanguages: (languages: string[]) => void;
  toggleLanguage: (language: string) => void;
  toggleCluster: (clusterId: number) => void;
  setSelectedClusters: (clusterIds: number[]) => void;
  clearClusterFilter: () => void;
  toggleStarList: (listId: number) => void;
  setSelectedStarLists: (listIds: number[]) => void;
  clearStarListFilter: () => void;
  clearFilters: () => void;
  setSyncing: (syncing: boolean) => void;
  setSyncStep: (step: string) => void;
}

const STORAGE_KEY_SETTINGS = 'nebula_graph_settings';

const defaultFilters: GraphFiltersState = {
  selectedClusters: new Set<number>(),
  selectedStarLists: new Set<number>(),
  searchQuery: '',
  timeRange: null,
  minStars: 0,
  languages: new Set<string>(),
};

const defaultSettings: GraphSettingsState = {
  showTrajectories: true,
  hqRendering: true,
  maxClusters: 8,
  minClusters: 3,
  relatedMinSemantic: 0.65,
};

const getInitialSettings = (): GraphSettingsState => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY_SETTINGS);
    if (!stored) return defaultSettings;
    return { ...defaultSettings, ...JSON.parse(stored) };
  } catch {
    return defaultSettings;
  }
};

export const useGraphStore = create<GraphUiState>((set) => ({
  selectedNode: null,
  filters: defaultFilters,
  settings: getInitialSettings(),
  syncing: false,
  syncStep: '',
  setSelectedNode: (node) => set({ selectedNode: node }),
  updateSettings: (settings) =>
    set((state) => {
      const updated = { ...state.settings, ...settings };
      localStorage.setItem(STORAGE_KEY_SETTINGS, JSON.stringify(updated));
      return { settings: updated };
    }),
  setSearchQuery: (query) =>
    set((state) => ({ filters: { ...state.filters, searchQuery: query } })),
  setTimeRange: (range) =>
    set((state) => ({ filters: { ...state.filters, timeRange: range } })),
  setMinStars: (min) =>
    set((state) => ({ filters: { ...state.filters, minStars: min } })),
  setSelectedLanguages: (languages) =>
    set((state) => ({
      filters: { ...state.filters, languages: new Set(languages) },
    })),
  toggleLanguage: (language) =>
    set((state) => {
      const next = new Set(state.filters.languages);
      if (next.has(language)) next.delete(language);
      else next.add(language);
      return { filters: { ...state.filters, languages: next } };
    }),
  toggleCluster: (clusterId) =>
    set((state) => {
      const next = new Set(state.filters.selectedClusters);
      if (next.has(clusterId)) next.delete(clusterId);
      else next.add(clusterId);
      return { filters: { ...state.filters, selectedClusters: next } };
    }),
  setSelectedClusters: (clusterIds) =>
    set((state) => ({
      filters: { ...state.filters, selectedClusters: new Set(clusterIds) },
    })),
  clearClusterFilter: () =>
    set((state) => ({
      filters: { ...state.filters, selectedClusters: new Set<number>() },
    })),
  toggleStarList: (listId) =>
    set((state) => {
      const next = new Set(state.filters.selectedStarLists);
      if (next.has(listId)) next.delete(listId);
      else next.add(listId);
      return { filters: { ...state.filters, selectedStarLists: next } };
    }),
  setSelectedStarLists: (listIds) =>
    set((state) => ({
      filters: { ...state.filters, selectedStarLists: new Set(listIds) },
    })),
  clearStarListFilter: () =>
    set((state) => ({
      filters: { ...state.filters, selectedStarLists: new Set<number>() },
    })),
  clearFilters: () => set({ filters: defaultFilters }),
  setSyncing: (syncing) => set({ syncing }),
  setSyncStep: (syncStep) => set({ syncStep }),
}));
