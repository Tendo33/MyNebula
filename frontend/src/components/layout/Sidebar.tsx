import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Network, Settings, Database, Github } from 'lucide-react';
import clsx from 'clsx';
import { useTranslation } from 'react-i18next';

export const Sidebar = () => {
  const { t } = useTranslation();

  const navItems = [
    { icon: LayoutDashboard, label: t('sidebar.dashboard'), path: '/' },
    { icon: Network, label: t('sidebar.graph'), path: '/graph' },
    { icon: Database, label: t('sidebar.data'), path: '/data' },
    { icon: Settings, label: t('sidebar.settings'), path: '/settings' },
  ];

  return (
    <aside className="fixed left-0 top-0 bottom-0 w-60 bg-bg-sidebar border-r border-border-light flex flex-col z-50">
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
    </aside>
  );
};
