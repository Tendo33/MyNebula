import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key,
  }),
}));

import { SearchInput } from '../SearchInput';

describe('SearchInput', () => {
  it('clears immediately and stays in sync with controlled updates', () => {
    const onSearch = vi.fn();
    const { rerender } = render(
      <SearchInput value="nebula" onSearch={onSearch} debounceMs={0} aria-label="Search" />
    );

    const input = screen.getByRole('searchbox', { name: 'Search' });
    expect(input).toHaveValue('nebula');

    fireEvent.click(screen.getByRole('button', { name: 'Clear' }));
    expect(onSearch).toHaveBeenCalledWith('');
    expect(input).toHaveValue('');

    rerender(<SearchInput value="cluster" onSearch={onSearch} debounceMs={0} aria-label="Search" />);
    expect(input).toHaveValue('cluster');
  });
});
