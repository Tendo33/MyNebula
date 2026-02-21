import { NavLink } from 'react-router-dom';
import { useEffect, useState } from 'react';
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
      setSidebarWidth(clampSidebarWidth(event.clientX));
    };

    const handlePointerUp = () => {
      setIsResizing(false);
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
    };
  }, [isResizing, isMobile]);

  return (
    <>
      {isMobile && (
        <button
          onClick={() => setMobileOpen((prev) => !prev)}
          className="fixed left-3 top-3 z-[70] inline-flex h-9 w-9 items-center justify-center rounded-md border border-border-light bg-white/95 text-text-main shadow-sm backdrop-blur-sm"
          aria-label={mobileOpen ? t('common.close') : t('common.open_menu', 'Open menu')}
        >
          {mobileOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
        </button>
      )}

      {isMobile && mobileOpen && (
        <button
          className="fixed inset-0 z-[55] bg-black/40"
          onClick={() => setMobileOpen(false)}
          aria-label={t('common.close')}
        />
      )}

      <aside
        className={clsx(
          'fixed left-0 top-0 bottom-0 bg-bg-sidebar/95 backdrop-blur-sm border-r border-border-light flex flex-col z-[60] transition-transform duration-200',
          isMobile ? (mobileOpen ? 'translate-x-0' : '-translate-x-full') : 'translate-x-0'
        )}
        style={{ width: `${isMobile ? Math.min(sidebarWidth, 300) : sidebarWidth}px` }}
      >
      {/* Header / Brand */}
      <a
        href="https://github.com/Tendo33/MyNebula"
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center gap-2 px-4 h-12 mt-2 cursor-pointer select-none transition-colors hover:bg-white mx-2 rounded-lg mb-2 border border-transparent hover:border-border-light"
      >
        <div className="w-5 h-5 flex items-center justify-center text-text-main">
           <Github className="w-4 h-4" />
        </div>
        <span className="text-sm font-semibold text-text-main truncate">
          {t('app.title')}
        </span>
      </a>

      {/* Navigation */}
      <nav className="flex-1 px-3 space-y-1">
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
                isActive
                  ? 'bg-white text-text-main font-medium shadow-sm border-border-light'
                  : 'text-text-muted border-transparent hover:bg-white/80 hover:text-text-main hover:border-border-light/70'
              )
            }
          >
            {({ isActive }) => (
              <>
                {isActive && (
                  <span className="absolute left-1 top-1/2 h-5 w-0.5 -translate-y-1/2 rounded-full bg-action-primary/60" />
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
          className="absolute top-0 right-0 h-full w-1.5 cursor-col-resize group"
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
