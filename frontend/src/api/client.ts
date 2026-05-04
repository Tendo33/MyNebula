import axios, { AxiosHeaders } from 'axios';
import type { AxiosError } from 'axios';

export const normalizeApiBaseOrigin = (value: string): string =>
  value.trim().replace(/\/+$/, '').replace(/\/api$/i, '');

export const getApiBaseUrl = (): string => {
	// 开发环境下始终使用相对路径，以便 Vite proxy 能够接管从而避免跨域和 Cookie 丢失问题
	if (import.meta.env.DEV) {
		return "/api";
	}

	// 首先检查环境变量
	const envUrl = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL;
	if (envUrl) {
		return `${normalizeApiBaseOrigin(envUrl)}/api`;
	}
	// 默认使用当前页面的 origin + /api，或者 localhost:8000
	if (typeof window !== "undefined") {
		return `${normalizeApiBaseOrigin(window.location.origin)}/api`;
	}
	return "http://localhost:8000/api";
};

export const API_BASE_URL = getApiBaseUrl();
const ADMIN_CSRF_COOKIE = 'nebula_admin_csrf';
const ADMIN_CSRF_HEADER = 'X-CSRF-Token';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
});

type UnauthorizedHandler = (error: AxiosError) => void;

let unauthorizedHandler: UnauthorizedHandler | null = null;

const getCookieValue = (cookieName: string): string | null => {
  if (typeof document === 'undefined') {
    return null;
  }
  const target = `${cookieName}=`;
  const match = document.cookie
    .split(';')
    .map((item) => item.trim())
    .find((item) => item.startsWith(target));
  return match ? decodeURIComponent(match.slice(target.length)) : null;
};

const isMutatingMethod = (method: string | undefined): boolean => {
  if (!method) {
    return false;
  }
  const normalized = method.toUpperCase();
  return normalized !== 'GET' && normalized !== 'HEAD' && normalized !== 'OPTIONS';
};

const shouldNotifyUnauthorized = (error: AxiosError): boolean => {
  if (error.response?.status !== 401) {
    return false;
  }

  const requestUrl = error.config?.url ?? '';
  if (requestUrl.includes('/v2/auth/login')) {
    return false;
  }

  return true;
};

apiClient.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (shouldNotifyUnauthorized(error) && unauthorizedHandler) {
      unauthorizedHandler(error);
    }
    return Promise.reject(error);
  }
);

apiClient.interceptors.request.use((config) => {
  if (!isMutatingMethod(config.method)) {
    return config;
  }
  const csrfToken = getCookieValue(ADMIN_CSRF_COOKIE);
  if (!csrfToken) {
    return config;
  }
  const headers = AxiosHeaders.from(config.headers ?? {});
  headers.set(ADMIN_CSRF_HEADER, csrfToken);
  config.headers = headers;
  return config;
});

export const setUnauthorizedHandler = (handler: UnauthorizedHandler | null): void => {
  unauthorizedHandler = handler;
};

export default apiClient;
