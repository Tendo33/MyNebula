import client from './client';

export interface AdminSessionResponse {
  authenticated: boolean;
  username: string;
}

interface LoginPayload {
  username: string;
  password: string;
}

export const loginAdmin = async (payload: LoginPayload): Promise<AdminSessionResponse> => {
  const response = await client.post<AdminSessionResponse>('/auth/login', payload);
  return response.data;
};

export const logoutAdmin = async (): Promise<AdminSessionResponse> => {
  const response = await client.post<AdminSessionResponse>('/auth/logout');
  return response.data;
};

export const getAdminSession = async (): Promise<AdminSessionResponse> => {
  const response = await client.get<AdminSessionResponse>('/auth/me');
  return response.data;
};
