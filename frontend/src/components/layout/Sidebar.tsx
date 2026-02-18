import { NavLink } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { LayoutDashboard, Network, Settings, Database, Github } from 'lucide-react';
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

  const navItems = [
    { icon: LayoutDashboard, label: t('sidebar.dashboard'), path: '/' },
    { icon: Network, label: t('sidebar.graph'), path: '/graph' },
    { icon: Database, label: t('sidebar.data'), path: '/data' },
    { icon: Settings, label: t('sidebar.settings'), path: '/settings' },
  ];

  useEffect(() => {
    document.documentElement.style.setProperty('--sidebar-width', `${sidebarWidth}px`);
    window.localStorage.setItem(SIDEBAR_WIDTH_KEY, String(sidebarWidth));
  }, [sidebarWidth]);

  useEffect(() => {
    if (!isResizing) return;

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
  }, [isResizing]);

  return (
    <aside
      className="fixed left-0 top-0 bottom-0 bg-bg-sidebar border-r border-border-light flex flex-col z-50"
      style={{ width: `${sidebarWidth}px` }}
    >
      {/* Header / Brand */}
      <a
        href="https://github.com/Tendo33/MyNebula"
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center gap-2 px-4 h-12 mt-2 cursor-pointer select-none transition-colors hover:bg-bg-hover mx-2 rounded-sm mb-2"
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
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-all duration-200 group select-none',
                isActive
                  ? 'bg-bg-hover text-text-main font-medium shadow-sm'
                  : 'text-text-muted hover:bg-bg-hover hover:text-text-main hover:shadow-sm'
              )
            }
          >
            <item.icon className={clsx(
              "w-4.5 h-4.5",
              ({ isActive }: { isActive: boolean }) => isActive ? "text-text-main" : "text-text-dim"
            )} />
            <span className="truncate">{item.label}</span>
          </NavLink>
        ))}
      </nav>

      {/* Optional User/Footer Area could go here */}

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
    </aside>
  );
};
