import React, { useState, useEffect, useCallback } from 'react';
import { Search, X } from 'lucide-react';
import { clsx } from 'clsx';
import { useTranslation } from 'react-i18next';

interface SearchInputProps extends Omit<React.InputHTMLAttributes<HTMLInputElement>, 'onChange'> {
  /** Callback when search value changes */
  onSearch?: (value: string) => void;
  /** Controlled value */
  value?: string;
  /** Debounce delay in ms (default: 300) */
  debounceMs?: number;
}

export const SearchInput: React.FC<SearchInputProps> = ({
  className,
  onSearch,
  value: controlledValue,
  placeholder,
  debounceMs = 300,
  ...props
}) => {
  const { t } = useTranslation();
  const [localValue, setLocalValue] = useState(controlledValue || '');

  // Sync with controlled value
  useEffect(() => {
    if (controlledValue !== undefined) {
      setLocalValue(controlledValue);
    }
  }, [controlledValue]);

  // Debounced search callback
  useEffect(() => {
    if (!onSearch) return;

    const timer = setTimeout(() => {
      onSearch(localValue);
    }, debounceMs);

    return () => clearTimeout(timer);
  }, [localValue, onSearch, debounceMs]);

  // Handle input change
  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    setLocalValue(e.target.value);
  }, []);

  // Handle clear
  const handleClear = useCallback(() => {
    setLocalValue('');
    if (onSearch) {
      onSearch('');
    }
  }, [onSearch]);

  return (
    <div className={clsx('relative group', className)}>
      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
        <Search className="h-4 w-4 text-text-dim group-focus-within:text-text-main transition-colors" />
      </div>
      <input
        type="text"
        value={localValue}
        onChange={handleChange}
        className="block w-full pl-9 pr-8 py-2 h-9 border border-border-light rounded-md leading-5 bg-bg-sidebar text-text-main placeholder-text-dim focus:outline-none focus:bg-white focus:ring-2 focus:ring-action-primary/20 focus:border-action-primary text-sm shadow-sm transition-all duration-200"
        placeholder={placeholder || t('dashboard.search_placeholder')}
        {...props}
      />
      {localValue && (
        <button
          type="button"
          onClick={handleClear}
          className="absolute inset-y-0 right-0 pr-3 flex items-center text-text-dim hover:text-text-main transition-colors"
        >
          <X className="h-4 w-4" />
        </button>
      )}
    </div>
  );
};
