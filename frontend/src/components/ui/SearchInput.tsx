import React from 'react';
import { Search } from 'lucide-react';
import { clsx } from 'clsx';
import { useTranslation } from 'react-i18next';

interface SearchInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  onSearch?: (value: string) => void;
}

export const SearchInput: React.FC<SearchInputProps> = ({ className, onSearch, onChange, ...props }) => {
  const { t } = useTranslation();
  return (
    <div className={clsx('relative group', className)}>
      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
        <Search className="h-4 w-4 text-text-dim group-focus-within:text-text-main transition-colors" />
      </div>
      <input
        type="text"
        className="block w-full pl-9 pr-3 py-2 h-9 border border-border-light rounded-md leading-5 bg-bg-sidebar text-text-main placeholder-text-dim focus:outline-none focus:bg-white focus:ring-2 focus:ring-action-primary/20 focus:border-action-primary text-sm shadow-sm transition-all duration-200"
        placeholder={t('dashboard.search_placeholder')}
        onChange={(e) => {
            if (onChange) onChange(e);
            if (onSearch) onSearch(e.target.value);
        }}
        {...props}
      />
    </div>
  );
};
