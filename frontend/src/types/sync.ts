export interface SyncStatusResponse {
  task_id: number | null;
  status: string;
  task_type: string;
  total_items: number;
  processed_items: number;
  failed_items: number;
  progress_percent: number;
  error_message: string | null;
  started_at: string | null;
  completed_at: string | null;
}
