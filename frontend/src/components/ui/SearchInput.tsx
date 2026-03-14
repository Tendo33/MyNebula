import React, { useState, useEffect, useCallback, useRef } from 'react';
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
  const skipNextDebounceRef = useRef(false);
  const computedAriaLabel =
    props['aria-label'] ??
    (props['aria-labelledby'] ? undefined : (placeholder || t('dashboard.search_placeholder')));

  // Sync with controlled value
  useEffect(() => {
    if (controlledValue !== undefined) {
      setLocalValue(controlledValue);
    }
  }, [controlledValue]);

  // Debounced search callback
  useEffect(() => {
    if (!onSearch) return;
    if (skipNextDebounceRef.current) {
      skipNextDebounceRef.current = false;
      return;
    }

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
    skipNextDebounceRef.current = true;
    setLocalValue('');
    if (onSearch) {
      onSearch('');
    }
  }, [onSearch]);

  return (
    <div className={clsx('relative group', className)}>
      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
        <Search className="h-4 w-4 text-text-dim group-focus-within:text-action-primary transition-colors" />
      </div>
      <input
        type="text"
        value={localValue}
        onChange={handleChange}
        className="block w-full pl-9 pr-8 py-2 h-9 border border-border-light rounded-lg leading-5 bg-bg-main/85 text-text-main placeholder-text-dim focus:outline-none focus:bg-bg-main focus:ring-2 focus:ring-action-primary/15 focus:border-action-primary text-sm shadow-sm hover:border-border-light dark:bg-dark-bg-main/85 dark:text-dark-text-main dark:border-dark-border dark:focus:bg-dark-bg-main"
        placeholder={placeholder || t('dashboard.search_placeholder')}
        aria-label={computedAriaLabel}
        {...props}
      />
      {localValue && (
        <button
          type="button"
          onClick={handleClear}
          aria-label={t('common.clear', 'Clear')}
          className="absolute inset-y-0 right-0 pr-3 flex items-center text-text-dim hover:text-text-main focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-action-primary/30 rounded"
        >
          <X className="h-4 w-4" />
        </button>
      )}
    </div>
  );
};
