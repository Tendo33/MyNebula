import type { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';

import { HeaderActions } from '@/components/layout/HeaderActions';
import { AppSidebar } from '@/components/layout/Sidebar';
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/ui/breadcrumb';
import { Separator } from '@/components/ui/separator';
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from '@/components/ui/sidebar';

const PAGE_LABEL_KEYS: Record<string, string> = {
  '/': 'sidebar.dashboard',
  '/graph': 'sidebar.graph',
  '/data': 'sidebar.data',
  '/settings': 'sidebar.settings',
};

export const AppShell = ({ children }: { children: ReactNode }) => {
  const { t } = useTranslation();
  const location = useLocation();
  const pageKey = PAGE_LABEL_KEYS[location.pathname];
  const pageLabel = pageKey ? t(pageKey) : t('errors.not_found');

  return (
    <SidebarProvider className="h-svh overflow-hidden">
      <a
        href="#main-content"
        className="fixed top-4 left-4 z-[100] -translate-y-24 rounded-md bg-primary px-4 py-2 text-primary-foreground focus:translate-y-0"
      >
        {t('common.skip_to_content')}
      </a>
      <AppSidebar />
      <SidebarInset id="main-content" className="min-h-0 overflow-hidden">
        <header className="flex h-12 shrink-0 items-center gap-2 border-b px-3">
          <SidebarTrigger />
          <Separator orientation="vertical" className="h-4" />
          <Breadcrumb>
            <BreadcrumbList>
              <BreadcrumbItem>
                <BreadcrumbLink render={<Link to="/" />}>{t('app.title')}</BreadcrumbLink>
              </BreadcrumbItem>
              <BreadcrumbSeparator />
              <BreadcrumbItem>
                <BreadcrumbPage>{pageLabel}</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>
          <div className="ml-auto">
            <HeaderActions />
          </div>
        </header>
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden">{children}</div>
      </SidebarInset>
    </SidebarProvider>
  );
};
