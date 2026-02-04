import React from 'react';
import { useTranslation } from 'react-i18next';
import { GraphNode } from '../../types';
import { X, Star, Code, ExternalLink, Sparkles } from 'lucide-react';

interface RepoDetailsPanelProps {
  node: GraphNode;
  onClose: () => void;
}

export const RepoDetailsPanel: React.FC<RepoDetailsPanelProps> = ({ node, onClose }) => {
  const { t } = useTranslation();

  return (
    <div className="absolute top-6 right-6 z-30 w-96 bg-white rounded-lg border border-border-light shadow-xl overflow-hidden animate-in fade-in slide-in-from-right-4 duration-300">
      {/* Header */}
      <div className="relative p-5 border-b border-border-light bg-bg-sidebar flex items-start justify-between">
        <div className="pr-8">
            <div className="flex items-center gap-2 mb-1.5">
                <a href={node.html_url} target="_blank" rel="noopener noreferrer" className="hover:underline decoration-text-muted underline-offset-4">
                    <h2 className="text-base font-semibold text-text-main line-clamp-1 leading-snug" title={node.name}>
                    {node.name}
                    </h2>
                </a>
            </div>
            <p className="text-sm text-text-muted line-clamp-2 leading-relaxed">
            {node.description || 'No description available'}
            </p>
        </div>
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 rounded-md text-text-dim hover:bg-bg-hover hover:text-text-main transition-colors"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Content */}
      <div className="p-5 space-y-6">

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-3">
            <div className="flex items-center gap-3 p-3 rounded-md bg-white border border-border-light shadow-sm">
                <div className="p-1.5 bg-orange-50 rounded text-orange-500">
                    <Star className="w-4 h-4" fill="currentColor" />
                </div>
                <div>
                    <span className="block text-lg font-semibold text-text-main leading-none">{node.stargazers_count?.toLocaleString() ?? 0}</span>
                    <span className="text-xs text-text-muted capitalize">{t('repoDetails.stars')}</span>
                </div>
            </div>
            <div className="flex items-center gap-3 p-3 rounded-md bg-white border border-border-light shadow-sm">
                 <div className="p-1.5 bg-blue-50 rounded text-blue-500">
                    <Code className="w-4 h-4" />
                </div>
                <div>
                    <span className="block text-lg font-semibold text-text-main leading-none truncate max-w-[100px]" title={node.language || 'Unknown'}>{node.language || 'N/A'}</span>
                    <span className="text-xs text-text-muted capitalize">{t('repoDetails.language')}</span>
                </div>
            </div>
        </div>

        {/* AI Summary */}
        <div className="space-y-2">
            <div className="flex items-center gap-2 text-xs font-semibold text-action-primary uppercase tracking-wider">
                <Sparkles className="w-3.5 h-3.5" />
                <span>AI Insight</span>
            </div>
            <div className="bg-bg-sidebar p-3 rounded-md border border-border-light/50">
                 <p className="text-sm text-text-main leading-relaxed">
                    {node.ai_summary || "No AI summary available for this repository yet. Sync data to generate insights."}
                </p>
            </div>
        </div>

        {/* Footer Actions */}
         <div>
             <a
                href={node.html_url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center justify-center w-full gap-2 px-4 py-2.5 bg-text-main hover:bg-black text-white text-sm font-medium rounded-md transition-all shadow hover:shadow-md"
             >
                 <ExternalLink className="w-4 h-4" />
                 <span>View on GitHub</span>
             </a>
         </div>
      </div>
    </div>
  );
};
