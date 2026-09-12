import { Link } from 'react-router-dom';

import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';

interface EmptyStateProps {
  title: string;
  description?: string;
  actionTo?: string;
  actionLabel?: string;
  onAction?: () => void;
  actionType?: 'link' | 'button';
}

export const EmptyState = ({
  title,
  description,
  actionTo,
  actionLabel,
  onAction,
  actionType = 'link',
}: EmptyStateProps) => {
  return (
    <div className="mx-auto flex min-h-[18rem] max-w-lg flex-col items-start justify-center py-12">
      <Separator className="mb-6 w-12" />
      <h2 className="font-heading text-[32px] font-semibold leading-10 tracking-[-1.28px] text-text-main">
        {title}
      </h2>
      {description ? (
        <p className="mt-3 max-w-prose text-sm leading-relaxed text-text-muted">{description}</p>
      ) : null}
      {actionLabel && actionType === 'button' && onAction ? (
        <Button type="button" onClick={onAction} className="mt-7">
          {actionLabel}
        </Button>
      ) : null}
      {actionLabel && actionType === 'link' && actionTo ? (
        <Button nativeButton={false} render={<Link to={actionTo} />} className="mt-7">
          {actionLabel}
        </Button>
      ) : null}
    </div>
  );
};
