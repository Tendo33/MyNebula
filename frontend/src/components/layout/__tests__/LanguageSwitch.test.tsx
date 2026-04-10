import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

const changeLanguage = vi.fn();

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback ?? key,
    i18n: {
      language: 'zh',
      changeLanguage,
    },
  }),
}));

import { LanguageSwitch } from '../LanguageSwitch';

describe('LanguageSwitch', () => {
  it('renders a readable zh label and toggles back to english', () => {
    render(<LanguageSwitch />);

    expect(screen.getByRole('button', { name: 'Switch language' })).toHaveTextContent('中文');

    fireEvent.click(screen.getByRole('button', { name: 'Switch language' }));

    expect(changeLanguage).toHaveBeenCalledWith('en');
  });
});
