import { Globe } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const LanguageSwitch = () => {
  const { t, i18n } = useTranslation();

  const toggleLanguage = () => {
    const nextLanguage = i18n.language === 'en' ? 'zh' : 'en';
    i18n.changeLanguage(nextLanguage);
  };

  return (
    <button
      onClick={toggleLanguage}
      className="h-11 sm:h-8 px-3 rounded-lg text-xs font-semibold border border-border-light bg-bg-main/90 hover:bg-bg-hover text-text-main min-w-[96px] text-center shadow-sm inline-flex items-center justify-center gap-1.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30 dark:bg-dark-bg-main/90 dark:text-dark-text-main dark:border-dark-border dark:hover:bg-dark-bg-sidebar/70"
      title={t('settings.language')}
    >
      <Globe className="w-3.5 h-3.5 text-text-dim" />
      <span>{i18n.language === 'en' ? 'EN' : '中'}</span>
    </button>
  );
};
