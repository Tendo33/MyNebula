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
    <div className={clsx('group relative', className)}>
      <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-4">
        <Search className="h-4 w-4 text-text-dim transition-colors group-focus-within:text-action-primary" />
      </div>
      <input
        type="search"
        value={localValue}
        onChange={handleChange}
        className="field-surface block h-11 w-full pl-11 pr-11 text-sm font-medium leading-5 placeholder:text-text-dim/90 sm:h-11"
        placeholder={placeholder || t('dashboard.search_placeholder')}
        aria-label={computedAriaLabel}
        {...props}
      />
      {localValue && (
        <button
          type="button"
          onClick={handleClear}
          aria-label={t('common.clear', 'Clear')}
          className="absolute inset-y-0 right-0 flex items-center pr-3.5 text-text-dim hover:text-text-main"
        >
          <X className="h-4 w-4" />
        </button>
      )}
    </div>
  );
};
