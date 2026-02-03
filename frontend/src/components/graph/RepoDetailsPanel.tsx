import React from 'react';
import { useTranslation } from 'react-i18next';
import { GraphNode } from '../../types';

interface RepoDetailsPanelProps {
  node: GraphNode;
  onClose: () => void;
}

export const RepoDetailsPanel: React.FC<RepoDetailsPanelProps> = ({ node, onClose }) => {
  const { t } = useTranslation();

  const formatNumber = (num: number): string => {
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'k';
    }
    return num.toString();
  };

  const formatDate = (dateStr?: string): string => {
    if (!dateStr) return t('repoDetails.unknown');
    return new Date(dateStr).toLocaleDateString();
  };

  return (
    <div className="absolute top-4 right-4 z-30 w-80 bg-nebula-surface/95 backdrop-blur-xl rounded-2xl border border-nebula-border/50 shadow-2xl shadow-black/50 overflow-hidden animate-fade-in">
      {/* Header */}
      <div className="relative p-4 border-b border-nebula-border/30 bg-gradient-to-r from-nebula-primary/10 to-transparent">
        <button
          onClick={onClose}
          className="absolute top-3 right-3 w-8 h-8 flex items-center justify-center rounded-lg bg-nebula-bg/50 hover:bg-nebula-bg text-nebula-text-muted hover:text-nebula-text-main transition-all duration-200"
          aria-label={t('repoDetails.close')}
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
        <h3 className="text-lg font-bold text-nebula-text-main pr-8 truncate">
          {node.name}
        </h3>
        <p className="text-xs text-nebula-text-muted truncate">
          {node.full_name}
        </p>
      </div>

      {/* Content */}
      <div className="p-4 space-y-4">
        {/* Description */}
        <div>
          <p className="text-sm text-nebula-text-main leading-relaxed">
            {node.description || t('repoDetails.no_description')}
          </p>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-3">
          {/* Stars */}
          <div className="flex items-center gap-2 p-3 rounded-xl bg-nebula-bg/50 border border-nebula-border/30">
            <span className="text-yellow-400">⭐</span>
            <div>
              <p className="text-xs text-nebula-text-muted">{t('repoDetails.stars')}</p>
              <p className="text-sm font-semibold text-nebula-text-main">
                {formatNumber(node.stargazers_count)}
              </p>
            </div>
          </div>

          {/* Language */}
          <div className="flex items-center gap-2 p-3 rounded-xl bg-nebula-bg/50 border border-nebula-border/30">
            <span className="text-nebula-primary">💻</span>
            <div>
              <p className="text-xs text-nebula-text-muted">{t('repoDetails.language')}</p>
              <p className="text-sm font-semibold text-nebula-text-main truncate">
                {node.language || t('repoDetails.unknown')}
              </p>
            </div>
          </div>
        </div>

        {/* Starred Date */}
        {node.starred_at && (
          <div className="flex items-center gap-2 text-sm text-nebula-text-muted">
            <span>📅</span>
            <span>{t('repoDetails.starred_at')}: {formatDate(node.starred_at)}</span>
          </div>
        )}

        {/* AI Summary */}
        {node.ai_summary && (
          <div className="p-3 rounded-xl bg-nebula-primary/5 border border-nebula-primary/20">
            <p className="text-xs text-nebula-primary mb-1 font-medium">✨ {t('repoDetails.ai_summary')}</p>
            <p className="text-sm text-nebula-text-main leading-relaxed">
              {node.ai_summary}
            </p>
          </div>
        )}

        {/* GitHub Link */}
        <a
          href={node.html_url}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center justify-center gap-2 w-full py-3 px-4 rounded-xl bg-nebula-primary/10 hover:bg-nebula-primary/20 text-nebula-primary font-medium text-sm transition-all duration-200 group"
        >
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
          </svg>
          {t('repoDetails.view_on_github')}
          <svg className="w-4 h-4 transition-transform group-hover:translate-x-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
          </svg>
        </a>
      </div>
    </div>
  );
};
