import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';

interface StatCardProps {
  label: string;
  value: string | number;
  subValue?: string;
  icon: LucideIcon;
  iconVariant?: 'blue' | 'purple' | 'emerald' | 'amber';
  linkText?: string;
  linkPath?: string;
  thumbnailUrl?: string;
  badge?: string;
  badgeVariant?: 'neutral' | 'success' | 'blue' | 'purple';
  isEvaluated?: boolean;
}

export const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  subValue,
  icon: Icon,
  iconVariant = 'blue',
  linkText,
  linkPath,
  thumbnailUrl,
  badge,
  isEvaluated = true,
}) => {
  const navigate = useNavigate();

  const getIconStyles = () => {
    switch (iconVariant) {
      case 'purple':
        return 'bg-purple-100 text-purple-600';
      case 'emerald':
        return 'bg-emerald-100 text-emerald-600';
      case 'amber':
        return 'bg-amber-100 text-amber-600';
      case 'blue':
      default:
        return 'bg-blue-100 text-blue-600';
    }
  };

  return (
    <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-xs hover:shadow-card transition-all duration-200 flex flex-col justify-between">
      <div>
        {/* Top Row: Circular Icon + Optional Right Thumbnail or Badge */}
        <div className="flex items-center justify-between mb-3">
          <div className={`w-9 h-9 rounded-full flex items-center justify-center ${getIconStyles()}`}>
            <Icon className="w-4 h-4" />
          </div>

          {thumbnailUrl ? (
            <div className="w-10 h-10 rounded-lg overflow-hidden border border-slate-200 bg-slate-950 shrink-0 shadow-2xs">
              <img
                src={thumbnailUrl}
                alt="Dataset thumbnail"
                className="w-full h-full object-cover filter contrast-125"
              />
            </div>
          ) : badge ? (
            <span className="w-6 h-6 rounded-full bg-purple-50 text-purple-600 border border-purple-200 text-[10px] font-bold flex items-center justify-center">
              {badge}
            </span>
          ) : null}
        </div>

        {/* Content */}
        <div>
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1">
            {label}
          </span>
          <h3
            className={`text-xl font-bold tracking-tight text-slate-900 ${
              !isEvaluated ? 'text-slate-400 italic text-base' : ''
            }`}
          >
            {value}
          </h3>
          {subValue && (
            <p className="text-xs text-slate-500 mt-1 font-normal leading-snug line-clamp-2">
              {subValue}
            </p>
          )}
        </div>
      </div>

      {/* Bottom Link with Arrow */}
      {linkText && linkPath && (
        <div className="mt-4 pt-3 border-t border-slate-100">
          <button
            onClick={() => navigate(linkPath)}
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-blue-600 hover:text-blue-700 transition-colors group cursor-pointer"
          >
            <span>{linkText}</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
          </button>
        </div>
      )}
    </div>
  );
};
