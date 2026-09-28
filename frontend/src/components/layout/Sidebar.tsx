import { useEffect } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { Database, Github, LayoutDashboard, Network, Settings } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  useSidebar,
} from '@/components/ui/sidebar';

const NAV_ITEMS = [
  { icon: LayoutDashboard, labelKey: 'sidebar.dashboard', path: '/' },
  { icon: Network, labelKey: 'sidebar.graph', path: '/graph' },
  { icon: Database, labelKey: 'sidebar.data', path: '/data' },
  { icon: Settings, labelKey: 'sidebar.settings', path: '/settings' },
] as const;

export const AppSidebar = () => {
  const { t } = useTranslation();
  const location = useLocation();
  const { setOpenMobile } = useSidebar();

  useEffect(() => {
    setOpenMobile(false);
  }, [location.pathname, setOpenMobile]);

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              size="lg"
              tooltip={t('app.title')}
              render={
                <a
                  href="https://github.com/Tendo33/MyNebula"
                  target="_blank"
                  rel="noopener noreferrer"
                />
              }
            >
              <Github />
              <span className="grid min-w-0 flex-1 text-left leading-tight">
                <span className="truncate font-semibold">{t('app.title')}</span>
                <span className="truncate text-xs text-muted-foreground">{t('sidebar.tagline')}</span>
              </span>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>{t('sidebar.navigation')}</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {NAV_ITEMS.map((item) => {
                const isActive =
                  item.path === '/'
                    ? location.pathname === '/'
                    : location.pathname === item.path;
                return (
                  <SidebarMenuItem key={item.path}>
                    <SidebarMenuButton
                      isActive={isActive}
                      tooltip={t(item.labelKey)}
                      render={<NavLink to={item.path} end={item.path === '/'} />}
                    >
                      <item.icon />
                      <span>{t(item.labelKey)}</span>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarRail />
    </Sidebar>
  );
};
