import { Globe } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';

export const LanguageSwitch = () => {
  const { t, i18n } = useTranslation();
  const currentLanguage = i18n.resolvedLanguage || i18n.language;
  const isZh = currentLanguage.startsWith('zh');

  const toggleLanguage = () => {
    void i18n.changeLanguage(isZh ? 'en' : 'zh');
  };

  return (
    <Tooltip>
      <TooltipTrigger render={<span className="inline-flex shrink-0" />}>
        <Button
          type="button"
          variant="outline"
          onClick={toggleLanguage}
          aria-label={t('settings.language', 'Switch language')}
        >
          <Globe data-icon="inline-start" />
          <span>{isZh ? '中文' : 'EN'}</span>
        </Button>
      </TooltipTrigger>
      <TooltipContent>{t('settings.language')}</TooltipContent>
    </Tooltip>
  );
};
