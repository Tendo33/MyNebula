import { NavLink } from 'react-router-dom';
import { useEffect, useRef, useState } from 'react';
import { LayoutDashboard, Network, Settings, Database, Github, Menu, X } from 'lucide-react';
import clsx from 'clsx';
import { useTranslation } from 'react-i18next';

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

  return (
    <>
      {isMobile && (
        <button
          type="button"
          onClick={() => setMobileOpen((prev) => !prev)}
          className="fixed left-3 top-3 z-[70] inline-flex h-11 w-11 items-center justify-center rounded-xl border border-border-light bg-bg-main/96 text-text-main shadow-sm backdrop-blur-md dark:border-dark-border dark:bg-dark-bg-main/96 dark:text-dark-text-main"
          aria-label={mobileOpen ? t('common.close') : t('common.open_menu', 'Open menu')}
        >
          {mobileOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
        </button>
      )}

      {isMobile && mobileOpen && (
        <button
          type="button"
          className="fixed inset-0 z-[55] bg-slate-950/28 backdrop-blur-[1px]"
          onClick={() => setMobileOpen(false)}
          aria-label={t('common.close')}
        />
      )}

      <aside
        className={clsx(
          'fixed bottom-0 left-0 top-0 z-[60] flex flex-col border-r border-border-light bg-bg-sidebar/95 backdrop-blur-md shadow-[0_24px_60px_-36px_rgba(15,23,42,0.45)] transition-transform duration-200 dark:border-dark-border dark:bg-dark-bg-sidebar/95',
          isMobile ? (mobileOpen ? 'translate-x-0' : '-translate-x-full') : 'translate-x-0'
        )}
        style={{ width: `${isMobile ? Math.min(sidebarWidth, 300) : sidebarWidth}px` }}
      >
      {/* Header / Brand */}
      <a
        href="https://github.com/Tendo33/MyNebula"
        target="_blank"
        rel="noopener noreferrer"
        className="mx-2 mb-2 mt-2 flex h-12 items-center gap-2 rounded-xl border border-transparent px-4 transition-colors hover:border-border-light hover:bg-bg-main/75 dark:hover:border-dark-border dark:hover:bg-dark-bg-main/65"
      >
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-bg-main/80 text-text-main shadow-sm dark:bg-dark-bg-main/80">
           <Github className="h-4 w-4" />
        </div>
        <span className="truncate text-sm font-semibold tracking-tight text-text-main">
          {t('app.title')}
        </span>
      </a>

      {/* Navigation */}
      <nav className="flex-1 space-y-1 px-3 py-2">
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
                'relative flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all duration-200 group select-none border',
                'min-h-[44px] rounded-xl border text-sm transition-all duration-200',
                isActive
                  ? 'border-border-light bg-bg-main/92 font-medium text-text-main shadow-sm dark:border-dark-border dark:bg-dark-bg-main/92 dark:text-dark-text-main'
                  : 'border-transparent text-text-muted hover:-translate-y-px hover:border-border-light/70 hover:bg-bg-main/72 hover:text-text-main dark:text-dark-text-main/70 dark:hover:border-dark-border dark:hover:bg-dark-bg-main/58 dark:hover:text-dark-text-main'
              )
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <span className="absolute left-1.5 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-full bg-action-primary/70" />
                )}
                <item.icon
                  className={clsx(
                    'h-4 w-4',
                    isActive ? 'text-text-main' : 'text-text-dim group-hover:text-text-main'
                  )}
                />
                <span className="truncate">{item.label}</span>
              </>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Optional User/Footer Area could go here */}

      {!isMobile && (
        <div
          role="separator"
          aria-orientation="vertical"
          aria-label="Resize sidebar"
          onPointerDown={(event) => {
            event.preventDefault();
            setIsResizing(true);
          }}
          className="group absolute right-0 top-0 h-full w-2 cursor-col-resize"
        >
          <div
            className={clsx(
              'absolute top-0 right-0 h-full w-px transition-colors',
              isResizing ? 'bg-action-primary/60' : 'bg-transparent group-hover:bg-border-light'
            )}
          />
        </div>
      )}
    </aside>
    </>
  );
};
