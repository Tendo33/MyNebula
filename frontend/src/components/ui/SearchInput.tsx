import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Search, X } from 'lucide-react';
import { useTranslation } from 'react-i18next';

import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupInput,
} from '@/components/ui/input-group';

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
    <InputGroup className={className}>
      <InputGroupAddon>
        <Search />
      </InputGroupAddon>
      <InputGroupInput
        type="search"
        value={localValue}
        onChange={handleChange}
        placeholder={placeholder || t('dashboard.search_placeholder')}
        aria-label={computedAriaLabel}
        {...props}
      />
      {localValue ? (
        <InputGroupAddon align="inline-end">
          <InputGroupButton
            size="icon-xs"
            onClick={handleClear}
            aria-label={t('common.clear', 'Clear')}
          >
            <X />
          </InputGroupButton>
        </InputGroupAddon>
      ) : null}
    </InputGroup>
  );
};
