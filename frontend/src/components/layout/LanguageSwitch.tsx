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
      className="h-8 px-3 rounded-md text-xs font-medium border border-border-light bg-white hover:bg-bg-hover text-text-main transition-colors min-w-[96px] text-center shadow-sm inline-flex items-center justify-center gap-1.5"
      title={t('settings.language')}
    >
      <Globe className="w-3.5 h-3.5" />
      <span>{i18n.language === 'en' ? 'EN' : '中'}</span>
    </button>
  );
};
