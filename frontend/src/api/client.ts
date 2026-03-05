import axios from 'axios';
import type { AxiosError } from 'axios';

const getApiBaseUrl = (): string => {
	// 开发环境下始终使用相对路径，以便 Vite proxy 能够接管从而避免跨域和 Cookie 丢失问题
	if (import.meta.env.DEV) {
		return "/api";
	}

	// 首先检查环境变量
	const envUrl = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL;
	if (envUrl) {
		return `${envUrl}/api`;
	}
	// 默认使用当前页面的 origin + /api，或者 localhost:8071
	if (typeof window !== "undefined") {
		return `${window.location.origin}/api`;
	}
	return "http://localhost:8071/api";
};

export const API_BASE_URL = getApiBaseUrl();

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true,
  headers: {
    'Content-Type': 'application/json',
  },
});

type UnauthorizedHandler = (error: AxiosError) => void;

let unauthorizedHandler: UnauthorizedHandler | null = null;

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

export const setUnauthorizedHandler = (handler: UnauthorizedHandler | null): void => {
  unauthorizedHandler = handler;
};

export default apiClient;
