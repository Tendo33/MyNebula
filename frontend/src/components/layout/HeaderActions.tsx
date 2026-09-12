import { type ReactNode, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { Search } from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Kbd, KbdGroup } from '@/components/ui/kbd';
import { LanguageSwitch } from './LanguageSwitch';

const openCommandPalette = () => {
  const isMac = typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform);
  window.dispatchEvent(
    new KeyboardEvent('keydown', {
      key: 'k',
      metaKey: isMac,
      ctrlKey: !isMac,
      bubbles: true,
    })
  );
};

export const CommandPaletteHint = () => {
  const { t } = useTranslation();
  const isMac = useMemo(
    () => typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform),
    []
  );

  return (
    <Button
      type="button"
      variant="outline"
      className="hidden max-w-[14rem] sm:inline-flex"
      onClick={openCommandPalette}
      aria-label={t('common.command_palette', 'Command Palette')}
    >
      <Search data-icon="inline-start" />
      <span className="truncate text-muted-foreground">
        {t('common.search_command', 'Search')}
      </span>
      <KbdGroup>
        <Kbd>{isMac ? '⌘' : 'Ctrl'}</Kbd>
        <Kbd>K</Kbd>
      </KbdGroup>
    </Button>
  );
};

export const HeaderActions = ({
  children,
  showSearchHint = true,
}: {
  children?: ReactNode;
  showSearchHint?: boolean;
}) => (
  <div className="flex flex-wrap items-center justify-end gap-2">
    {showSearchHint ? <CommandPaletteHint /> : null}
    <LanguageSwitch />
    {children}
  </div>
);
