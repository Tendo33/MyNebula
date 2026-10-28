// Applied before the bundle loads: without this a dark-mode user gets a
// white flash on every navigation while React boots.
(function () {
  try {
    var stored = localStorage.getItem('nebula_theme') || 'system';
    var dark =
      stored === 'dark' ||
      (stored === 'system' &&
        window.matchMedia &&
        window.matchMedia('(prefers-color-scheme: dark)').matches);
    if (dark) {
      document.documentElement.classList.add('dark');
      document.documentElement.style.colorScheme = 'dark';
    }
  } catch {
    /* no stored preference; fall through to light */
  }
})();
