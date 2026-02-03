import axios from 'axios';

// 优先使用环境变量，否则根据当前页面地址推断后端地址
const getApiBaseUrl = (): string => {
  // 首先检查环境变量
  const envUrl = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL;
  if (envUrl) {
    return `${envUrl}/api`;
  }
  // 默认使用 localhost:8000 (uvicorn 默认端口)
  return 'http://localhost:8000/api';
};

export const API_BASE_URL = getApiBaseUrl();

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add interceptor for JWT token if needed
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default apiClient;
