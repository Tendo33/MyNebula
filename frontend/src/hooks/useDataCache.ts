import { useCallback, useRef } from "react";

// ============================================================================
// Types
// ============================================================================

interface CacheEntry<T> {
	data: T;
	timestamp: number;
	expiresAt: number;
}

interface CacheOptions {
	/** Cache duration in milliseconds (default: 5 minutes) */
	ttl?: number;
	/** Storage key prefix */
	keyPrefix?: string;
	/** Whether to use localStorage for persistence */
	persist?: boolean;
}

interface UseCacheReturn<T> {
	/** Get cached data */
	get: (key: string) => T | null;
	/** Set cached data */
	set: (key: string, data: T) => void;
	/** Remove cached data */
	remove: (key: string) => void;
	/** Clear all cached data */
	clear: () => void;
	/** Check if key exists and is not expired */
	has: (key: string) => boolean;
	/** Preload data into cache */
	preload: (key: string, fetcher: () => Promise<T>) => Promise<T>;
}

// ============================================================================
// Constants
// ============================================================================

const DEFAULT_TTL = 5 * 60 * 1000; // 5 minutes
const DEFAULT_PREFIX = "nebula_cache_";

// ============================================================================
// Hook
// ============================================================================

export function useDataCache<T = any>(options: CacheOptions = {}): UseCacheReturn<T> {
	const { ttl = DEFAULT_TTL, keyPrefix = DEFAULT_PREFIX, persist = true } = options;

	const memoryCache = useRef<Map<string, CacheEntry<T>>>(new Map());

	// Build full key with prefix
	const buildKey = useCallback((key: string) => `${keyPrefix}${key}`, [keyPrefix]);

	// Get from localStorage
	const getFromStorage = useCallback(
		(fullKey: string): CacheEntry<T> | null => {
			if (!persist) return null;

			try {
				const stored = localStorage.getItem(fullKey);
				if (stored) {
					return JSON.parse(stored);
				}
			} catch (e) {
				console.warn("Failed to read from localStorage:", e);
			}
			return null;
		},
		[persist],
	);

	// Set to localStorage
	const setToStorage = useCallback(
		(fullKey: string, entry: CacheEntry<T>) => {
			if (!persist) return;

			try {
				localStorage.setItem(fullKey, JSON.stringify(entry));
			} catch (e) {
				console.warn("Failed to write to localStorage:", e);
			}
		},
		[persist],
	);

	// Remove from localStorage
	const removeFromStorage = useCallback(
		(fullKey: string) => {
			if (!persist) return;

			try {
				localStorage.removeItem(fullKey);
			} catch (e) {
				console.warn("Failed to remove from localStorage:", e);
			}
		},
		[persist],
	);

	// Check if entry is expired
	const isExpired = useCallback((entry: CacheEntry<T>): boolean => {
		return Date.now() > entry.expiresAt;
	}, []);

	// Get cached data
	const get = useCallback(
		(key: string): T | null => {
			const fullKey = buildKey(key);

			// Check memory cache first
			const memoryEntry = memoryCache.current.get(fullKey);
			if (memoryEntry && !isExpired(memoryEntry)) {
				return memoryEntry.data;
			}

			// Check localStorage
			const storageEntry = getFromStorage(fullKey);
			if (storageEntry && !isExpired(storageEntry)) {
				// Restore to memory cache
				memoryCache.current.set(fullKey, storageEntry);
				return storageEntry.data;
			}

			// Expired or not found
			if (memoryEntry) {
				memoryCache.current.delete(fullKey);
			}
			if (storageEntry) {
				removeFromStorage(fullKey);
			}

			return null;
		},
		[buildKey, isExpired, getFromStorage, removeFromStorage],
	);

	// Set cached data
	const set = useCallback(
		(key: string, data: T) => {
			const fullKey = buildKey(key);
			const now = Date.now();

			const entry: CacheEntry<T> = {
				data,
				timestamp: now,
				expiresAt: now + ttl,
			};

			memoryCache.current.set(fullKey, entry);
			setToStorage(fullKey, entry);
		},
		[buildKey, ttl, setToStorage],
	);

	// Remove cached data
	const remove = useCallback(
		(key: string) => {
			const fullKey = buildKey(key);
			memoryCache.current.delete(fullKey);
			removeFromStorage(fullKey);
		},
		[buildKey, removeFromStorage],
	);

	// Clear all cached data with prefix
	const clear = useCallback(() => {
		// Clear memory cache
		memoryCache.current.clear();

		// Clear localStorage
		if (persist) {
			try {
				const keys = Object.keys(localStorage);
				keys.forEach((key) => {
					if (key.startsWith(keyPrefix)) {
						localStorage.removeItem(key);
					}
				});
			} catch (e) {
				console.warn("Failed to clear localStorage:", e);
			}
		}
	}, [keyPrefix, persist]);

	// Check if key exists and is not expired
	const has = useCallback(
		(key: string): boolean => {
			return get(key) !== null;
		},
		[get],
	);

	// Preload data into cache
	const preload = useCallback(
		async (key: string, fetcher: () => Promise<T>): Promise<T> => {
			// Check cache first
			const cached = get(key);
			if (cached !== null) {
				return cached;
			}

			// Fetch and cache
			const data = await fetcher();
			set(key, data);
			return data;
		},
		[get, set],
	);

	return { get, set, remove, clear, has, preload };
}

// ============================================================================
// Specialized Cache for Graph Data
// ============================================================================

export function useGraphCache() {
	const cache = useDataCache({
		ttl: 10 * 60 * 1000, // 10 minutes for graph data
		keyPrefix: "nebula_graph_",
		persist: true,
	});

	return {
		...cache,
		// Specific methods for graph data
		getGraphData: () => cache.get("main"),
		setGraphData: (data: any) => cache.set("main", data),
		getTimelineData: () => cache.get("timeline"),
		setTimelineData: (data: any) => cache.set("timeline", data),
		invalidate: () => {
			cache.remove("main");
			cache.remove("timeline");
		},
	};
}

// ============================================================================
// Specialized Cache for Node Details
// ============================================================================

export function useNodeDetailsCache() {
	const cache = useDataCache({
		ttl: 30 * 60 * 1000, // 30 minutes for node details
		keyPrefix: "nebula_node_",
		persist: true,
	});

	return {
		...cache,
		getNode: (nodeId: number) => cache.get(`${nodeId}`),
		setNode: (nodeId: number, data: any) => cache.set(`${nodeId}`, data),
	};
}

export default useDataCache;
