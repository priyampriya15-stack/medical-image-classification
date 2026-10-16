import React from 'react';

export const CardSkeleton: React.FC<{ className?: string }> = ({ className = 'h-32' }) => {
  return (
    <div className={`bg-white rounded-2xl p-5 border border-slate-200/80 shadow-card animate-pulse ${className}`}>
      <div className="flex justify-between items-start mb-4">
        <div className="space-y-2">
          <div className="h-3 w-20 bg-slate-200 rounded" />
          <div className="h-6 w-32 bg-slate-200 rounded" />
        </div>
        <div className="w-10 h-10 bg-slate-200 rounded-xl" />
      </div>
      <div className="h-3 w-28 bg-slate-100 rounded" />
    </div>
  );
};

export const ChartSkeleton: React.FC<{ height?: string }> = ({ height = 'h-72' }) => {
  return (
    <div className={`bg-white rounded-2xl p-6 border border-slate-200/80 shadow-card animate-pulse ${height} flex flex-col justify-between`}>
      <div className="flex justify-between items-center mb-6">
        <div className="h-4 w-44 bg-slate-200 rounded" />
        <div className="flex gap-2">
          <div className="h-7 w-16 bg-slate-200 rounded-lg" />
          <div className="h-7 w-16 bg-slate-200 rounded-lg" />
        </div>
      </div>
      <div className="flex items-end justify-between gap-4 h-44 px-4">
        {[60, 85, 45, 95, 75, 90, 80].map((h, i) => (
          <div key={i} className="flex-1 flex gap-1.5 items-end h-full">
            <div className="w-1/2 bg-slate-200 rounded-t" style={{ height: `${h * 0.7}%` }} />
            <div className="w-1/2 bg-blue-100 rounded-t" style={{ height: `${h * 0.9}%` }} />
          </div>
        ))}
      </div>
      <div className="h-3 w-full bg-slate-100 rounded mt-4" />
    </div>
  );
};
