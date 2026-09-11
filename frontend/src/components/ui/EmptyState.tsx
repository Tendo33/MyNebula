import { Link } from 'react-router-dom';

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
      <div className="mb-6 h-px w-12 bg-border-light" />
      <h2 className="font-heading text-[32px] font-semibold leading-10 tracking-[-1.28px] text-text-main">
        {title}
      </h2>
      {description ? (
        <p className="mt-3 max-w-prose text-sm leading-relaxed text-text-muted">{description}</p>
      ) : null}
      {actionLabel && actionType === 'button' && onAction ? (
        <button type="button" onClick={onAction} className="header-action mt-7">
          {actionLabel}
        </button>
      ) : null}
      {actionLabel && actionType === 'link' && actionTo ? (
        <Link to={actionTo} className="header-action mt-7">
          {actionLabel}
        </Link>
      ) : null}
    </div>
  );
};
