import { NavLink } from 'react-router-dom';
import { useEffect, useRef, useState } from 'react';
import { LayoutDashboard, Network, Settings, Database, Github, Menu } from 'lucide-react';
import clsx from 'clsx';
import { useTranslation } from 'react-i18next';

import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTitle, SheetTrigger } from '@/components/ui/sheet';

const SIDEBAR_MIN_WIDTH = 200;
const SIDEBAR_MAX_WIDTH = 420;
const SIDEBAR_DEFAULT_WIDTH = 240;
const SIDEBAR_WIDTH_KEY = 'mynebula.sidebar.width';

const clampSidebarWidth = (width: number) =>
  Math.min(SIDEBAR_MAX_WIDTH, Math.max(SIDEBAR_MIN_WIDTH, width));

export const Sidebar = () => {
  const { t } = useTranslation();
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    if (typeof window === 'undefined') {
      return SIDEBAR_DEFAULT_WIDTH;
    }

    const storedWidth = Number(window.localStorage.getItem(SIDEBAR_WIDTH_KEY));
    return Number.isFinite(storedWidth)
      ? clampSidebarWidth(storedWidth)
      : SIDEBAR_DEFAULT_WIDTH;
  });
  const [isResizing, setIsResizing] = useState(false);
  const [isMobile, setIsMobile] = useState(
    typeof window !== 'undefined' ? window.innerWidth < 1024 : false
  );
  const [mobileOpen, setMobileOpen] = useState(false);
  const resizeRafRef = useRef<number | null>(null);
  const pendingWidthRef = useRef<number | null>(null);

  const navItems = [
    { icon: LayoutDashboard, label: t('sidebar.dashboard'), path: '/' },
    { icon: Network, label: t('sidebar.graph'), path: '/graph' },
    { icon: Database, label: t('sidebar.data'), path: '/data' },
    { icon: Settings, label: t('sidebar.settings'), path: '/settings' },
  ];

  useEffect(() => {
    const effectiveWidth = isMobile ? 0 : sidebarWidth;
    document.documentElement.style.setProperty('--sidebar-width', `${effectiveWidth}px`);
    window.localStorage.setItem(SIDEBAR_WIDTH_KEY, String(sidebarWidth));
  }, [sidebarWidth, isMobile]);

  useEffect(() => {
    const handleResize = () => {
      const nextIsMobile = window.innerWidth < 1024;
      setIsMobile(nextIsMobile);
      if (!nextIsMobile) {
        setMobileOpen(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    if (!isResizing || isMobile) return;

    const handlePointerMove = (event: PointerEvent) => {
      pendingWidthRef.current = event.clientX;
      if (resizeRafRef.current === null) {
        resizeRafRef.current = window.requestAnimationFrame(() => {
          if (pendingWidthRef.current !== null) {
            setSidebarWidth(clampSidebarWidth(pendingWidthRef.current));
          }
          resizeRafRef.current = null;
        });
      }
    };

    const handlePointerUp = () => {
      setIsResizing(false);
      if (resizeRafRef.current !== null) {
        window.cancelAnimationFrame(resizeRafRef.current);
        resizeRafRef.current = null;
      }
    };

    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', handlePointerUp);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    return () => {
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', handlePointerUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      if (resizeRafRef.current !== null) {
        window.cancelAnimationFrame(resizeRafRef.current);
        resizeRafRef.current = null;
      }
    };
  }, [isResizing, isMobile]);

  const chrome = (
    <>
      <a
        href="https://github.com/Tendo33/MyNebula"
        target="_blank"
        rel="noopener noreferrer"
        className="mx-3 mb-2 mt-3 flex min-h-12 items-center gap-2.5 rounded-md px-2 py-2 hover:bg-bg-hover"
      >
        <div className="flex h-6 w-6 items-center justify-center rounded-md border border-border-light bg-bg-elevated text-text-main">
          <Github className="h-3.5 w-3.5" />
        </div>
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold tracking-[-0.28px] text-text-main">
            {t('app.title')}
          </div>
          <div className="truncate font-mono text-xs text-text-dim">
            {t('sidebar.tagline')}
          </div>
        </div>
      </a>

      <nav className="flex flex-1 flex-col gap-0.5 px-3 py-1">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            onClick={() => {
              if (isMobile) {
                setMobileOpen(false);
              }
            }}
            className={({ isActive }) =>
              clsx(
                'group relative flex min-h-10 items-center gap-2 rounded-md px-2 text-sm transition-colors',
                isActive
                  ? 'bg-bg-hover font-medium text-text-main'
                  : 'text-text-muted hover:bg-bg-hover hover:text-text-main'
              )
            }
          >
            {({ isActive }) => (
              <>
                <item.icon
                  className={clsx(
                    'h-4 w-4',
                    isActive ? 'text-text-main' : 'text-text-dim group-hover:text-text-main'
                  )}
                />
                <span className={clsx('truncate font-medium', isActive && 'font-semibold')}>
                  {item.label}
                </span>
              </>
            )}
          </NavLink>
        ))}
      </nav>
    </>
  );

  if (isMobile) {
    return (
      <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
        <SheetTrigger
          render={
            <Button
              type="button"
              variant="outline"
              size="icon"
              className="fixed left-3 top-3 z-[70]"
            />
          }
          aria-label={t('common.open_menu', 'Open menu')}
        >
          <Menu />
        </SheetTrigger>
        <SheetContent
          side="left"
          className="w-[min(100%,18.75rem)] gap-0 bg-sidebar p-0"
        >
          <SheetTitle className="sr-only">{t('common.open_menu', 'Open menu')}</SheetTitle>
          {chrome}
        </SheetContent>
      </Sheet>
    );
  }

  return (
    <aside
      className="fixed bottom-0 left-0 top-0 z-[60] flex flex-col border-r border-border-light bg-bg-sidebar dark:border-dark-border dark:bg-dark-bg-sidebar"
      style={{ width: `${sidebarWidth}px` }}
    >
      {chrome}
      <div
        role="separator"
        tabIndex={0}
        aria-orientation="vertical"
        aria-label="Resize sidebar"
        aria-valuemin={SIDEBAR_MIN_WIDTH}
        aria-valuemax={SIDEBAR_MAX_WIDTH}
        aria-valuenow={sidebarWidth}
        onKeyDown={(event) => {
          const increment = event.shiftKey ? 40 : 10;
          if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
            event.preventDefault();
            setSidebarWidth((current) =>
              clampSidebarWidth(current + (event.key === 'ArrowRight' ? increment : -increment))
            );
          } else if (event.key === 'Home') {
            event.preventDefault();
            setSidebarWidth(SIDEBAR_MIN_WIDTH);
          } else if (event.key === 'End') {
            event.preventDefault();
            setSidebarWidth(SIDEBAR_MAX_WIDTH);
          }
        }}
        onPointerDown={(event) => {
          event.preventDefault();
          setIsResizing(true);
        }}
        className="group absolute right-0 top-0 h-full w-2 cursor-col-resize focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary"
      >
        <div
          className={clsx(
            'absolute top-0 right-0 h-full w-px transition-colors',
            isResizing ? 'bg-action-primary/60' : 'bg-transparent group-hover:bg-border-light'
          )}
        />
      </div>
    </aside>
  );
};
