import React from 'react';
import { Search } from 'lucide-react';
import { clsx } from 'clsx';

interface SearchInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  onSearch?: (value: string) => void;
}

export const SearchInput: React.FC<SearchInputProps> = ({ className, onSearch, onChange, ...props }) => {
  return (
    <div className={clsx('relative group', className)}>
      <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
        <Search className="h-4 w-4 text-nebula-text-muted group-focus-within:text-nebula-primary transition-colors" />
      </div>
      <input
        type="text"
        className="block w-full pl-10 pr-3 py-2 border border-nebula-border rounded-xl leading-5 bg-nebula-surface/50 text-nebula-text-main placeholder-nebula-text-dim focus:outline-none focus:ring-1 focus:ring-nebula-primary focus:border-nebula-primary sm:text-sm shadow-sm transition-all duration-200"
        placeholder="Search repositories via semantics..."
        onChange={(e) => {
            if (onChange) onChange(e);
            if (onSearch) onSearch(e.target.value);
        }}
        {...props}
      />
      {/* Glow effect on focus */}
      <div className="absolute inset-0 rounded-xl ring-2 ring-nebula-primary/20 opacity-0 group-focus-within:opacity-100 transition-opacity pointer-events-none" />
    </div>
  );
};
