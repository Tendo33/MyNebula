import { Globe } from 'lucide-react';
import { useTranslation } from 'react-i18next';

export const LanguageSwitch = () => {
  const { t, i18n } = useTranslation();
  const currentLanguage = i18n.resolvedLanguage || i18n.language;
  const isZh = currentLanguage.startsWith('zh');

  const toggleLanguage = () => {
    void i18n.changeLanguage(isZh ? 'en' : 'zh');
  };

  return (
    <button
      type="button"
      onClick={toggleLanguage}
      className="header-action group h-11 min-h-0 shrink-0 px-3 text-xs font-semibold tracking-[0.08em]"
      title={t('settings.language')}
      aria-label={t('settings.language', 'Switch language')}
    >
      <Globe className="h-3.5 w-3.5 text-text-dim transition-colors group-hover:text-text-main" />
      <span>{isZh ? '中文' : 'EN'}</span>
    </button>
  );
};
