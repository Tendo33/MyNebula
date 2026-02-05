import { Sidebar } from '../components/layout/Sidebar';
import { useTranslation } from 'react-i18next';
import { API_BASE_URL } from '../api/client';
import { Globe, Zap, Eye, Server, Shield, Clock, RefreshCw, Database, AlertTriangle, X, Loader2 } from 'lucide-react';
import { clsx } from 'clsx';
import { useState, useEffect, useCallback } from 'react';
import {
  getSchedule,
  updateSchedule,
  getSyncInfo,
  triggerFullRefresh,
  formatLastRunTime,
  formatNextRunTime,
  getStatusDisplay,
  type ScheduleConfig,
  type ScheduleResponse,
  type SyncInfoResponse,
} from '../api/schedule';

const Settings = () => {
  const { t, i18n } = useTranslation();
  const [hqRendering, setHqRendering] = useState(true);
  const [showTrajectories, setShowTrajectories] = useState(false);

  // Schedule state
  const [schedule, setSchedule] = useState<ScheduleResponse | null>(null);
  const [syncInfo, setSyncInfo] = useState<SyncInfoResponse | null>(null);
  const [scheduleLoading, setScheduleLoading] = useState(false);
  const [refreshLoading, setRefreshLoading] = useState(false);
  const [showConfirmDialog, setShowConfirmDialog] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Load schedule and sync info
  const loadScheduleData = useCallback(async () => {
    try {
      setError(null);
      const [scheduleData, infoData] = await Promise.all([
        getSchedule(),
        getSyncInfo(),
      ]);
      setSchedule(scheduleData);
      setSyncInfo(infoData);
    } catch (err) {
      setError('加载调度数据失败');
      console.error('Failed to load schedule data:', err);
    }
  }, []);

  useEffect(() => {
    loadScheduleData();
  }, [loadScheduleData]);

  // Handle schedule toggle
  const handleScheduleToggle = async () => {
    if (!schedule) return;
    setScheduleLoading(true);
    try {
      const newConfig: ScheduleConfig = {
        is_enabled: !schedule.is_enabled,
        schedule_hour: schedule.schedule_hour,
        schedule_minute: schedule.schedule_minute,
        timezone: schedule.timezone,
      };
      const updated = await updateSchedule(newConfig);
      setSchedule(updated);
    } catch (err) {
      setError('更新调度设置失败');
      console.error('Failed to update schedule:', err);
    } finally {
      setScheduleLoading(false);
    }
  };

  // Handle time change
  const handleTimeChange = async (hour: number, minute: number) => {
    if (!schedule) return;
    setScheduleLoading(true);
    try {
      const newConfig: ScheduleConfig = {
        is_enabled: schedule.is_enabled,
        schedule_hour: hour,
        schedule_minute: minute,
        timezone: schedule.timezone,
      };
      const updated = await updateSchedule(newConfig);
      setSchedule(updated);
    } catch (err) {
      setError('更新调度时间失败');
      console.error('Failed to update schedule time:', err);
    } finally {
      setScheduleLoading(false);
    }
  };

  // Handle full refresh
  const handleFullRefresh = async () => {
    setShowConfirmDialog(false);
    setRefreshLoading(true);
    try {
      await triggerFullRefresh();
      // Reload data to show updated status
      await loadScheduleData();
    } catch (err) {
      setError('启动全量刷新失败');
      console.error('Failed to trigger full refresh:', err);
    } finally {
      setRefreshLoading(false);
    }
  };

  const toggleLanguage = () => {
    const newLang = i18n.language === 'en' ? 'zh' : 'en';
    i18n.changeLanguage(newLang);
  };

  return (
    <div className="flex min-h-screen bg-bg-main text-text-main">
      <Sidebar />
      <main className="flex-1 ml-60 flex flex-col min-w-0">
         {/* Header */}
         <header className="flex items-center h-14 px-8 border-b border-border-light sticky top-0 bg-bg-main/95 backdrop-blur-sm z-40">
              <h1 className="text-base font-semibold text-text-main select-none tracking-tight">
                  {t('settings.title')}
              </h1>
         </header>

        <div className="max-w-3xl px-8 py-10 space-y-12">
            {/* Appearance */}
            <section>
                <h2 className="text-sm font-semibold text-text-muted uppercase tracking-wider mb-4 px-2 select-none">{t('settings.appearance')}</h2>
                <div className="space-y-2">
                     {/* Language */}
                     <div className="flex items-center justify-between p-3 rounded-md hover:bg-bg-hover transition-all group">
                        <div className="flex items-center gap-3">
                            <div className="p-2 rounded-md bg-bg-sidebar group-hover:bg-white transition-colors">
                                <Globe className="w-5 h-5 text-text-muted group-hover:text-text-main" />
                            </div>
                            <div className="flex flex-col">
                                <span className="text-sm font-medium text-text-main">{t('settings.language')}</span>
                                <span className="text-xs text-text-muted">Change the language of the interface</span>
                            </div>
                        </div>
                        <button
                            onClick={toggleLanguage}
                            className="h-8 px-3 rounded-md text-sm font-medium border border-border-light bg-white hover:bg-bg-hover text-text-main transition-colors min-w-[80px] text-center shadow-sm"
                        >
                            {i18n.language === 'en' ? 'English' : '中文'}
                        </button>
                    </div>

                    {/* Rendering Toggle */}
                    <div className="flex items-center justify-between p-3 rounded-md hover:bg-bg-hover transition-all group cursor-pointer" onClick={() => setHqRendering(!hqRendering)}>
                        <div className="flex items-center gap-3">
                            <div className="p-2 rounded-md bg-bg-sidebar group-hover:bg-white transition-colors">
                                <Zap className="w-5 h-5 text-text-muted group-hover:text-text-main" />
                            </div>
                             <div className="flex flex-col">
                                <span className="text-sm font-medium text-text-main">{t('settings.hq_rendering')}</span>
                                <span className="text-xs text-text-muted">Enable high-quality visual effects</span>
                            </div>
                        </div>
                        <button
                            className={clsx(
                                "relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none",
                                hqRendering ? "bg-black" : "bg-gray-300"
                            )}
                        >
                            <span className={clsx(
                                "inline-block h-5 w-5 transform rounded-full bg-white transition duration-200 ease-in-out shadow-sm",
                                hqRendering ? "translate-x-5" : "translate-x-0.5"
                            )} />
                        </button>
                    </div>

                    {/* Trajectories Toggle */}
                    <div className="flex items-center justify-between p-3 rounded-md hover:bg-bg-hover transition-all group cursor-pointer" onClick={() => setShowTrajectories(!showTrajectories)}>
                        <div className="flex items-center gap-3">
                             <div className="p-2 rounded-md bg-bg-sidebar group-hover:bg-white transition-colors">
                                <Eye className="w-5 h-5 text-text-muted group-hover:text-text-main" />
                            </div>
                             <div className="flex flex-col">
                                <span className="text-sm font-medium text-text-main">{t('settings.show_trajectories')}</span>
                                <span className="text-xs text-text-muted">Show connection paths between nodes</span>
                            </div>
                        </div>
                         <button
                            className={clsx(
                                "relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none",
                                showTrajectories ? "bg-black" : "bg-gray-300"
                            )}
                        >
                            <span className={clsx(
                                "inline-block h-5 w-5 transform rounded-full bg-white transition duration-200 ease-in-out shadow-sm",
                                showTrajectories ? "translate-x-5" : "translate-x-0.5"
                            )} />
                        </button>
                    </div>
                </div>
            </section>

             <hr className="border-t border-border-light" />

             {/* API Configuration */}
             <section>
                <h2 className="text-sm font-semibold text-text-muted uppercase tracking-wider mb-4 px-2 select-none">{t('settings.connection')}</h2>
                <div className="space-y-2">
                    <div className="p-3">
                         <div className="flex items-center gap-2 mb-3">
                             <Server className="w-4 h-4 text-text-muted" />
                             <label className="text-sm font-medium text-text-main">{t('settings.api_endpoint')}</label>
                         </div>
                        <input
                            type="text"
                            value={API_BASE_URL}
                            readOnly
                            className="w-full bg-bg-sidebar/50 border border-border-light rounded-md px-3 py-2 text-sm text-text-muted font-mono"
                        />
                    </div>

                    <div className="flex items-center justify-between p-3 rounded-md hover:bg-bg-hover transition-colors">
                         <div className="flex items-center gap-2">
                             <Shield className="w-4 h-4 text-text-muted" />
                             <label className="text-sm font-medium text-text-main">{t('settings.github_token_status')}</label>
                         </div>
                         <div className="flex items-center gap-2 text-green-700 text-sm font-medium bg-green-50 px-3 py-1 rounded-full border border-green-200">
                             <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse" />
                             {t('settings.connected')}
                         </div>
                    </div>
                </div>
            </section>

            <hr className="border-t border-border-light" />

            {/* Scheduled Sync */}
            <section>
                <h2 className="text-sm font-semibold text-text-muted uppercase tracking-wider mb-4 px-2 select-none">定时同步</h2>

                {error && (
                  <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md text-sm text-red-700 flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4" />
                    {error}
                    <button onClick={() => setError(null)} className="ml-auto hover:text-red-900">
                      <X className="w-4 h-4" />
                    </button>
                  </div>
                )}

                <div className="space-y-2">
                    {/* Enable Toggle */}
                    <div
                      className="flex items-center justify-between p-3 rounded-md hover:bg-bg-hover transition-all group cursor-pointer"
                      onClick={!scheduleLoading ? handleScheduleToggle : undefined}
                    >
                        <div className="flex items-center gap-3">
                            <div className="p-2 rounded-md bg-bg-sidebar group-hover:bg-white transition-colors">
                                <Clock className="w-5 h-5 text-text-muted group-hover:text-text-main" />
                            </div>
                            <div className="flex flex-col">
                                <span className="text-sm font-medium text-text-main">启用定时同步</span>
                                <span className="text-xs text-text-muted">每天自动增量更新 Star 数据</span>
                            </div>
                        </div>
                        <button
                            disabled={scheduleLoading}
                            className={clsx(
                                "relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none",
                                schedule?.is_enabled ? "bg-black" : "bg-gray-300",
                                scheduleLoading && "opacity-50 cursor-not-allowed"
                            )}
                        >
                            {scheduleLoading ? (
                              <Loader2 className="w-4 h-4 text-white mx-auto animate-spin" />
                            ) : (
                              <span className={clsx(
                                  "inline-block h-5 w-5 transform rounded-full bg-white transition duration-200 ease-in-out shadow-sm",
                                  schedule?.is_enabled ? "translate-x-5" : "translate-x-0.5"
                              )} />
                            )}
                        </button>
                    </div>

                    {/* Time Selection (only show when enabled) */}
                    {schedule?.is_enabled && (
                      <div className="p-3 pl-14">
                        <div className="flex items-center gap-4">
                          <div className="flex items-center gap-2">
                            <label className="text-sm text-text-muted">执行时间:</label>
                            <select
                              value={schedule.schedule_hour}
                              onChange={(e) => handleTimeChange(Number(e.target.value), schedule.schedule_minute)}
                              disabled={scheduleLoading}
                              className="bg-bg-sidebar border border-border-light rounded-md px-3 py-1.5 text-sm text-text-main focus:outline-none focus:ring-1 focus:ring-black"
                            >
                              {Array.from({ length: 24 }, (_, i) => (
                                <option key={i} value={i}>{i.toString().padStart(2, '0')}</option>
                              ))}
                            </select>
                            <span className="text-text-muted">:</span>
                            <select
                              value={schedule.schedule_minute}
                              onChange={(e) => handleTimeChange(schedule.schedule_hour, Number(e.target.value))}
                              disabled={scheduleLoading}
                              className="bg-bg-sidebar border border-border-light rounded-md px-3 py-1.5 text-sm text-text-main focus:outline-none focus:ring-1 focus:ring-black"
                            >
                              {[0, 15, 30, 45].map((m) => (
                                <option key={m} value={m}>{m.toString().padStart(2, '0')}</option>
                              ))}
                            </select>
                          </div>
                          <span className="text-xs text-text-muted">({schedule.timezone})</span>
                        </div>
                      </div>
                    )}

                    {/* Schedule Status */}
                    <div className="p-3 pl-14 space-y-1">
                      <div className="flex items-center gap-2 text-xs text-text-muted">
                        <span>上次运行:</span>
                        <span className="text-text-main">
                          {schedule ? formatLastRunTime(schedule.last_run_at) : '加载中...'}
                        </span>
                        {schedule?.last_run_status && (
                          <span className={clsx("font-medium", getStatusDisplay(schedule.last_run_status).color)}>
                            ({getStatusDisplay(schedule.last_run_status).text})
                          </span>
                        )}
                      </div>
                      {schedule?.is_enabled && schedule.next_run_at && (
                        <div className="flex items-center gap-2 text-xs text-text-muted">
                          <span>下次运行:</span>
                          <span className="text-text-main">
                            {formatNextRunTime(schedule.next_run_at, schedule.timezone)}
                          </span>
                        </div>
                      )}
                    </div>
                </div>
            </section>

            <hr className="border-t border-border-light" />

            {/* Data Management */}
            <section>
                <h2 className="text-sm font-semibold text-text-muted uppercase tracking-wider mb-4 px-2 select-none">数据管理</h2>
                <div className="space-y-2">
                    {/* Repository Statistics */}
                    <div className="p-3">
                        <div className="flex items-center gap-2 mb-3">
                            <Database className="w-4 h-4 text-text-muted" />
                            <label className="text-sm font-medium text-text-main">仓库统计</label>
                        </div>
                        <div className="grid grid-cols-2 gap-4 bg-bg-sidebar/50 rounded-md p-4">
                          <div>
                            <div className="text-2xl font-semibold text-text-main">{syncInfo?.total_repos ?? '-'}</div>
                            <div className="text-xs text-text-muted">总仓库数</div>
                          </div>
                          <div>
                            <div className="text-2xl font-semibold text-text-main">{syncInfo?.synced_repos ?? '-'}</div>
                            <div className="text-xs text-text-muted">已同步</div>
                          </div>
                          <div>
                            <div className="text-2xl font-semibold text-text-main">{syncInfo?.embedded_repos ?? '-'}</div>
                            <div className="text-xs text-text-muted">已向量化</div>
                          </div>
                          <div>
                            <div className="text-2xl font-semibold text-text-main">{syncInfo?.summarized_repos ?? '-'}</div>
                            <div className="text-xs text-text-muted">已生成摘要</div>
                          </div>
                        </div>
                        {syncInfo?.last_sync_at && (
                          <div className="mt-2 text-xs text-text-muted">
                            上次同步: {new Date(syncInfo.last_sync_at).toLocaleString('zh-CN')}
                          </div>
                        )}
                    </div>

                    {/* Full Refresh Button */}
                    <div className="p-3">
                        <div className="flex items-center gap-2 mb-3">
                            <RefreshCw className="w-4 h-4 text-text-muted" />
                            <label className="text-sm font-medium text-text-main">全量刷新</label>
                        </div>
                        <p className="text-xs text-text-muted mb-3">
                          重新获取所有 Star 数据，重新生成 AI 摘要和向量嵌入。这是一个耗时操作，会消耗 API 额度。
                        </p>
                        <button
                          onClick={() => setShowConfirmDialog(true)}
                          disabled={refreshLoading}
                          className={clsx(
                            "flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors",
                            "bg-red-50 border border-red-200 text-red-700 hover:bg-red-100",
                            refreshLoading && "opacity-50 cursor-not-allowed"
                          )}
                        >
                          {refreshLoading ? (
                            <>
                              <Loader2 className="w-4 h-4 animate-spin" />
                              处理中...
                            </>
                          ) : (
                            <>
                              <RefreshCw className="w-4 h-4" />
                              执行全量刷新
                            </>
                          )}
                        </button>
                    </div>
                </div>
            </section>
        </div>

        {/* Confirm Dialog */}
        {showConfirmDialog && (
          <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
            <div className="bg-white rounded-lg shadow-xl max-w-md w-full mx-4 p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-red-100 rounded-full">
                  <AlertTriangle className="w-6 h-6 text-red-600" />
                </div>
                <h3 className="text-lg font-semibold text-gray-900">确认全量刷新?</h3>
              </div>
              <p className="text-sm text-gray-600 mb-6">
                此操作将重置并重新处理所有 {syncInfo?.total_repos ?? 0} 个仓库的数据，包括：
              </p>
              <ul className="text-sm text-gray-600 mb-6 space-y-1 list-disc list-inside">
                <li>重新从 GitHub 获取所有 Star 数据</li>
                <li>重新生成所有 AI 摘要和标签</li>
                <li>重新计算所有向量嵌入</li>
                <li>重新执行聚类分析</li>
              </ul>
              <p className="text-xs text-amber-600 bg-amber-50 p-2 rounded mb-6">
                ⚠️ 这可能需要较长时间并消耗大量 API 额度
              </p>
              <div className="flex justify-end gap-3">
                <button
                  onClick={() => setShowConfirmDialog(false)}
                  className="px-4 py-2 text-sm font-medium text-gray-700 bg-gray-100 hover:bg-gray-200 rounded-md transition-colors"
                >
                  取消
                </button>
                <button
                  onClick={handleFullRefresh}
                  className="px-4 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-md transition-colors"
                >
                  确认刷新
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default Settings;
