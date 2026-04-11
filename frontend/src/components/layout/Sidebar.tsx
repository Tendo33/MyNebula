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
          className="header-action fixed left-3 top-3 z-[70] h-11 w-11 px-0"
          aria-label={mobileOpen ? t('common.close') : t('common.open_menu', 'Open menu')}
        >
          {mobileOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
        </button>
      )}

      {isMobile && mobileOpen && (
        <button
          type="button"
          className="fixed inset-0 z-[55] bg-slate-950/34 backdrop-blur-[2px]"
          onClick={() => setMobileOpen(false)}
          aria-label={t('common.close')}
        />
      )}

      <aside
        className={clsx(
          'fixed bottom-0 left-0 top-0 z-[60] flex flex-col border-r border-border-light/85 bg-bg-sidebar/88 backdrop-blur-xl shadow-[0_24px_64px_-36px_rgba(28,34,46,0.28)] transition-transform duration-200 dark:border-dark-border/90 dark:bg-dark-bg-sidebar/88',
          isMobile ? (mobileOpen ? 'translate-x-0' : '-translate-x-full') : 'translate-x-0'
        )}
        style={{ width: `${isMobile ? Math.min(sidebarWidth, 300) : sidebarWidth}px` }}
      >
        <a
          href="https://github.com/Tendo33/MyNebula"
          target="_blank"
          rel="noopener noreferrer"
          className="panel-subtle mx-3 mb-3 mt-3 flex min-h-[4.5rem] items-center gap-3 px-4 py-3 hover:bg-bg-main/78 dark:hover:bg-dark-bg-main/72"
        >
          <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-bg-main/85 text-text-main shadow-sm dark:bg-dark-bg-main/85">
            <Github className="h-4 w-4" />
          </div>
          <div className="min-w-0">
            <div className="font-heading truncate text-sm font-semibold text-text-main">
              {t('app.title')}
            </div>
            <div className="truncate text-[11px] font-medium uppercase tracking-[0.18em] text-text-dim">
              {t('sidebar.tagline')}
            </div>
          </div>
        </a>

        <nav className="flex-1 space-y-1.5 px-3 py-1">
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
                'group relative flex min-h-[48px] items-center gap-3 rounded-2xl border px-3 py-2.5 text-sm transition-all duration-200',
                isActive
                  ? 'border-border-light/95 bg-bg-main/94 text-text-main shadow-sm dark:border-dark-border/90 dark:bg-dark-bg-main/90 dark:text-dark-text-main'
                  : 'border-transparent text-text-muted hover:-translate-y-px hover:border-border-light/70 hover:bg-bg-main/68 hover:text-text-main dark:text-dark-text-main/70 dark:hover:border-dark-border/85 dark:hover:bg-dark-bg-main/58 dark:hover:text-dark-text-main'
              )
            }
          >
            {({ isActive }) => (
              <>
                <span
                  className={clsx(
                    'h-2 w-2 rounded-full transition-all',
                    isActive
                      ? 'scale-100 bg-action-primary shadow-[0_0_0_4px_rgba(45,89,200,0.12)]'
                      : 'scale-75 bg-border-light group-hover:bg-action-primary/45 dark:bg-dark-border'
                  )}
                />
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
