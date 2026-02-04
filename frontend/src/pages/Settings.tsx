import { Sidebar } from '../components/layout/Sidebar';
import { useTranslation } from 'react-i18next';
import { API_BASE_URL } from '../api/client';
import { Globe, Zap, Eye, Server, Shield } from 'lucide-react';
import { clsx } from 'clsx';
import { useState } from 'react';

const Settings = () => {
  const { t, i18n } = useTranslation();
  const [hqRendering, setHqRendering] = useState(true);
  const [showTrajectories, setShowTrajectories] = useState(false);

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
        </div>
      </main>
    </div>
  );
};

export default Settings;
