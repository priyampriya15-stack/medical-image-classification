import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Image as ImageIcon, ArrowRight } from 'lucide-react';
import type { DatasetSample } from '../../types';
import { EmptyState } from '../common/EmptyState';

interface DatasetPreviewCardProps {
  samples: DatasetSample[];
  loading?: boolean;
}

export const DatasetPreviewCard: React.FC<DatasetPreviewCardProps> = ({
  samples,
  loading = false,
}) => {
  const navigate = useNavigate();

  return (
    <div className="bg-white rounded-2xl p-5 sm:p-6 border border-slate-200/80 shadow-xs flex flex-col justify-between">
      <div>
        {/* Header */}
        <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-100">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-blue-50 border border-blue-100 flex items-center justify-center text-blue-600 shrink-0">
              <ImageIcon className="w-4 h-4" />
            </div>
            <h3 className="text-sm font-bold text-slate-900 tracking-tight">
              Dataset Preview
            </h3>
          </div>

          <button
            onClick={() => navigate('/dataset')}
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-blue-600 hover:text-blue-700 transition-colors group cursor-pointer"
          >
            <span>View All</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
          </button>
        </div>

        {/* 5 Radiographs in a Row */}
        {loading ? (
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 animate-pulse">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-28 bg-slate-100 rounded-xl" />
            ))}
          </div>
        ) : samples.length === 0 ? (
          <EmptyState
            compact
            title="No dataset samples"
            description="Dataset samples are not currently available."
          />
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
            {samples.slice(0, 5).map((sample) => {
              const isNormal = sample.classLabel === 'Normal';

              return (
                <div key={sample.id} className="flex flex-col items-center">
                  <div className="w-full h-24 sm:h-28 rounded-xl overflow-hidden bg-slate-950 border border-slate-200 shadow-2xs group relative">
                    <img
                      src={sample.imageUrl}
                      alt={`Radiograph ${sample.id}`}
                      className="w-full h-full object-cover filter contrast-125 brightness-95 group-hover:scale-105 transition-transform duration-300"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-slate-950/40 via-transparent to-transparent pointer-events-none" />
                  </div>

                  {/* Centered Pill Badge Below Image */}
                  <div className="mt-2">
                    <span
                      className={`text-[11px] font-semibold px-2.5 py-0.5 rounded-full border ${
                        isNormal
                          ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                          : 'bg-rose-50 text-rose-700 border-rose-200'
                      }`}
                    >
                      {sample.classLabel}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
};
