import React from 'react';
import { ShieldAlert, Info } from 'lucide-react';

interface DisclaimerBannerProps {
  variant?: 'banner' | 'card' | 'inline';
  className?: string;
}

export const DisclaimerBanner: React.FC<DisclaimerBannerProps> = ({
  variant = 'banner',
  className = '',
}) => {
  if (variant === 'inline') {
    return (
      <div className={`flex items-center gap-2 text-xs text-slate-500 ${className}`}>
        <Info className="w-3.5 h-3.5 text-blue-500 shrink-0" />
        <span>
          Educational & Research Platform. Experimental transfer-learning outputs only; not for clinical diagnosis.
        </span>
      </div>
    );
  }

  if (variant === 'card') {
    return (
      <div className={`p-4 rounded-xl border border-amber-200/70 bg-amber-50/50 flex items-start gap-3 ${className}`}>
        <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
        <div className="text-xs text-amber-900 leading-relaxed">
          <span className="font-semibold text-amber-950">Academic Safety Disclaimer: </span>
          This software is engineered strictly for comparative machine learning research and academic benchmarking.
          All predictions and activation maps are experimental algorithmic outputs and do not constitute clinical medical evaluations or verified medical advice.
        </div>
      </div>
    );
  }

  return (
    <div className={`px-4 py-2 bg-slate-900 text-slate-300 border-b border-slate-800 text-xs flex items-center justify-between ${className}`}>
      <div className="flex items-center gap-2 max-w-4xl mx-auto text-center sm:text-left">
        <ShieldAlert className="w-3.5 h-3.5 text-cyan-400 shrink-0 hidden sm:block" />
        <p>
          <span className="font-semibold text-white">Educational & Research Use Only: </span>
          Comparative Study of ResNet50 vs EfficientNetB0 on PneumoniaMNIST. Not a clinically validated diagnostic system.
        </p>
      </div>
    </div>
  );
};
