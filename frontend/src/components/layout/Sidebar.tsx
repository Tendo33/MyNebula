import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Network, Settings, Github, Database } from 'lucide-react';
import clsx from 'clsx';

export const Sidebar = () => {
  const navItems = [
    { icon: LayoutDashboard, label: 'Dashboard', path: '/' },
    { icon: Network, label: 'Graph', path: '/graph' },
    { icon: Database, label: 'Data', path: '/data' },
    { icon: Settings, label: 'Settings', path: '/settings' },
  ];

  return (
    <aside className="fixed left-4 top-4 bottom-4 w-64 bg-nebula-surface/80 backdrop-blur-xl border border-nebula-border rounded-2xl flex flex-col p-4 shadow-2xl z-50">
      <div className="flex items-center gap-3 px-2 mb-8 mt-2">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-nebula-primary to-nebula-secondary flex items-center justify-center shadow-lg shadow-nebula-primary/20">
          <Github className="w-5 h-5 text-white" />
        </div>
        <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-nebula-primary to-nebula-secondary">
          MyNebula
        </h1>
      </div>

      <nav className="flex-1 space-y-2">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all duration-200 group',
                isActive
                  ? 'bg-nebula-primary/10 text-nebula-primary shadow-[0_0_15px_rgba(0,255,255,0.15)] ring-1 ring-nebula-primary/20'
                  : 'text-nebula-text-muted hover:text-nebula-text-main hover:bg-nebula-surfaceHighlight'
              )
            }
          >
            <item.icon className="w-5 h-5 transition-transform group-hover:scale-110" />
            <span className="font-medium">{item.label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="px-3 py-4 border-t border-nebula-border mt-auto">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-nebula-surfaceHighlight ring-2 ring-nebula-border" />
          <div className="flex-1 overflow-hidden">
            <p className="text-sm font-medium text-nebula-text-main truncate">User</p>
            <p className="text-xs text-nebula-text-dim truncate">user@github.com</p>
          </div>
        </div>
      </div>
    </aside>
  );
};
