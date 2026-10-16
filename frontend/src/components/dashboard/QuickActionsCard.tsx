import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Zap,
  Upload,
  Play,
  Scale,
  BarChart2,
  Flame,
  ChevronRight,
} from 'lucide-react';

export const QuickActionsCard: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-xs">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4 pb-3 border-b border-slate-100">
        <div className="w-8 h-8 rounded-lg bg-blue-50 border border-blue-100 flex items-center justify-center text-blue-600 shrink-0">
          <Zap className="w-4 h-4 fill-blue-600" />
        </div>
        <h3 className="text-sm font-bold text-slate-900 tracking-tight">
          Quick Actions
        </h3>
      </div>

      <div className="space-y-2.5">
        {/* 1. Primary Highlighted Action: Upload Medical Image */}
        <button
          onClick={() => navigate('/classifier')}
          className="w-full text-left p-3.5 rounded-xl bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white shadow-xs hover:shadow-md transition-all duration-200 flex items-center justify-between group cursor-pointer"
        >
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-white/20 flex items-center justify-center text-white shrink-0">
              <Upload className="w-4 h-4" />
            </div>
            <div>
              <div className="text-xs font-bold leading-tight">
                Upload Medical Image
              </div>
              <div className="text-[11px] text-blue-100 font-normal leading-tight mt-0.5">
                JPG, JPEG, PNG (Max 10MB)
              </div>
            </div>
          </div>
          <ChevronRight className="w-4 h-4 text-white/80 group-hover:translate-x-0.5 transition-transform" />
        </button>

        {/* 2. Start Training */}
        <button
          onClick={() => navigate('/training')}
          className="w-full text-left p-3 rounded-xl border border-slate-200/80 hover:border-blue-300 bg-white hover:bg-slate-50 transition-all flex items-center justify-between group cursor-pointer"
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-slate-100 flex items-center justify-center text-slate-700 group-hover:bg-blue-50 group-hover:text-blue-600 transition-colors shrink-0">
              <Play className="w-3.5 h-3.5 fill-current" />
            </div>
            <div>
              <div className="text-xs font-bold text-slate-900 group-hover:text-blue-600 transition-colors">
                Start Training
              </div>
              <div className="text-[11px] text-slate-500">
                Train a model on the dataset
              </div>
            </div>
          </div>
          <ChevronRight className="w-4 h-4 text-slate-400 group-hover:text-blue-600 group-hover:translate-x-0.5 transition-all" />
        </button>

        {/* 3. Compare Models */}
        <button
          onClick={() => navigate('/comparison')}
          className="w-full text-left p-3 rounded-xl border border-slate-200/80 hover:border-blue-300 bg-white hover:bg-slate-50 transition-all flex items-center justify-between group cursor-pointer"
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-slate-100 flex items-center justify-center text-slate-700 group-hover:bg-blue-50 group-hover:text-blue-600 transition-colors shrink-0">
              <Scale className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="text-xs font-bold text-slate-900 group-hover:text-blue-600 transition-colors">
                Compare Models
              </div>
              <div className="text-[11px] text-slate-500">
                View ResNet50 vs EfficientNetB0
              </div>
            </div>
          </div>
          <ChevronRight className="w-4 h-4 text-slate-400 group-hover:text-blue-600 group-hover:translate-x-0.5 transition-all" />
        </button>

        {/* 4. View Analytics */}
        <button
          onClick={() => navigate('/analytics')}
          className="w-full text-left p-3 rounded-xl border border-slate-200/80 hover:border-blue-300 bg-white hover:bg-slate-50 transition-all flex items-center justify-between group cursor-pointer"
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-slate-100 flex items-center justify-center text-slate-700 group-hover:bg-blue-50 group-hover:text-blue-600 transition-colors shrink-0">
              <BarChart2 className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="text-xs font-bold text-slate-900 group-hover:text-blue-600 transition-colors">
                View Analytics
              </div>
              <div className="text-[11px] text-slate-500">
                Check training metrics & charts
              </div>
            </div>
          </div>
          <ChevronRight className="w-4 h-4 text-slate-400 group-hover:text-blue-600 group-hover:translate-x-0.5 transition-all" />
        </button>

        {/* 5. Grad-CAM */}
        <button
          onClick={() => navigate('/gradcam')}
          className="w-full text-left p-3 rounded-xl border border-slate-200/80 hover:border-blue-300 bg-white hover:bg-slate-50 transition-all flex items-center justify-between group cursor-pointer"
        >
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-slate-100 flex items-center justify-center text-slate-700 group-hover:bg-amber-50 group-hover:text-amber-600 transition-colors shrink-0">
              <Flame className="w-3.5 h-3.5" />
            </div>
            <div>
              <div className="text-xs font-bold text-slate-900 group-hover:text-amber-600 transition-colors">
                Grad-CAM
              </div>
              <div className="text-[11px] text-slate-500">
                Visualize model attention
              </div>
            </div>
          </div>
          <ChevronRight className="w-4 h-4 text-slate-400 group-hover:text-amber-600 group-hover:translate-x-0.5 transition-all" />
        </button>
      </div>
    </div>
  );
};
