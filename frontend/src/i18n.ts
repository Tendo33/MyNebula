import en from './locales/en.json';
import zh from './locales/zh.json';

type Dict = { [key: string]: string | Dict };
export type Lang = 'en' | 'zh';

const DICTS: Record<Lang, Dict> = { en, zh };
// Same key the previous i18next detector used, so a saved choice carries over.
const STORAGE_KEY = 'i18nextLng';
const listeners = new Set<() => void>();

const detect = (): Lang => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) return stored.startsWith('zh') ? 'zh' : 'en';
  } catch {
    // Storage blocked: fall back to the browser language.
  }
  return navigator.language?.startsWith('zh') ? 'zh' : 'en';
};

let current: Lang = detect();
document.documentElement.lang = current;

const lookup = (dict: Dict, key: string): string | undefined => {
  let node: string | Dict | undefined = dict;
  for (const part of key.split('.')) {
    if (typeof node !== 'object') return undefined;
    node = node[part];
  }
  return typeof node === 'string' ? node : undefined;
};

export const t = (key: string, params: Record<string, string | number> = {}): string => {
  const template = lookup(DICTS[current], key) ?? lookup(DICTS.en, key) ?? key;
  return template.replace(/\{\{(\w+)\}\}/g, (_, name) => String(params[name] ?? ''));
};

export const getLang = (): Lang => current;

export const setLang = (lang: Lang): void => {
  current = lang;
  document.documentElement.lang = lang;
  try {
    localStorage.setItem(STORAGE_KEY, lang);
  } catch {
    // Choice stays for this session only.
  }
  listeners.forEach((fn) => fn());
};

export const onLangChange = (fn: () => void): (() => void) => {
  listeners.add(fn);
  return () => listeners.delete(fn);
};
