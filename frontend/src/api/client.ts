import axios from 'axios';
import type { AxiosError } from 'axios';

// 优先使用环境变量，否则根据当前页面地址推断后端地址
const getApiBaseUrl = (): string => {
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
};;

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
  if (requestUrl.includes('/auth/login')) {
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
