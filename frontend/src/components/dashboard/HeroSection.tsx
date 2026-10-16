import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Cpu, Layers } from 'lucide-react';

export const HeroSection: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-blue-50/90 via-indigo-50/50 to-white border border-blue-100/80 p-6 sm:p-8 shadow-xs">
      {/* Soft background radial aura */}
      <div className="absolute top-1/2 right-1/4 -translate-y-1/2 w-72 h-72 rounded-full bg-blue-200/30 blur-3xl pointer-events-none" />

      <div className="relative z-10 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6 sm:gap-8">
        {/* Left Narrative */}
        <div className="max-w-xl">
          <span className="text-[11px] font-bold uppercase tracking-wider text-blue-600 block mb-2">
            Medical Image Classification
          </span>

          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-extrabold text-slate-900 tracking-tight leading-tight mb-2">
            Smarter Models. Better Insights.
          </h1>

          <p className="text-xs sm:text-sm text-slate-600 leading-relaxed mb-6 font-normal">
            Comparative study of ResNet50 and EfficientNetB0 using transfer learning for medical image classification.
          </p>

          <button
            onClick={() => navigate('/comparison')}
            className="inline-flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs sm:text-sm shadow-xs hover:shadow-md transition-all duration-200 cursor-pointer"
          >
            <span>Explore Project</span>
            <ArrowRight className="w-4 h-4" />
          </button>
        </div>

        {/* Right Graphic: Tilted Chest X-Ray + Floating Model Badges */}
        <div className="flex items-center justify-center lg:justify-end gap-4 shrink-0">
          {/* Tilted Radiograph Card */}
          <div className="relative group">
            <div className="w-44 h-32 sm:w-52 sm:h-36 rounded-2xl overflow-hidden bg-slate-950 border-2 border-white shadow-xl -rotate-2 group-hover:rotate-0 transition-transform duration-300 relative">
              {/* Chest Radiograph Illustration */}
              <svg
                viewBox="0 0 200 140"
                className="w-full h-full object-cover filter contrast-125"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <rect width="200" height="140" fill="#080C14" />
                {/* Thoracic spine */}
                <line x1="100" y1="10" x2="100" y2="130" stroke="#334155" strokeWidth="4" strokeDasharray="3 3" />
                {/* Clavicles */}
                <path d="M40 30 Q100 40 100 42 Q100 40 160 30" stroke="#64748B" strokeWidth="3" fill="none" opacity="0.7" />
                {/* Left & Right Lung fields (translucent radiolucent areas) */}
                <path
                  d="M45 42 C45 35 75 35 88 45 C95 55 95 105 85 115 C72 125 50 115 45 100 Z"
                  fill="url(#lungGlow)"
                  opacity="0.85"
                />
                <path
                  d="M155 42 C155 35 125 35 112 45 C105 55 105 105 115 115 C128 125 150 115 155 100 Z"
                  fill="url(#lungGlow)"
                  opacity="0.85"
                />
                {/* Rib cage arcs */}
                <path d="M50 50 Q100 65 150 50" stroke="#475569" strokeWidth="2" opacity="0.6" />
                <path d="M46 68 Q100 85 154 68" stroke="#475569" strokeWidth="2" opacity="0.6" />
                <path d="M44 86 Q100 105 156 86" stroke="#475569" strokeWidth="2" opacity="0.6" />
                <path d="M45 104 Q100 122 155 104" stroke="#475569" strokeWidth="2" opacity="0.6" />
                {/* Cardiac silhouette */}
                <ellipse cx="92" cy="85" rx="24" ry="20" fill="#1E293B" opacity="0.8" />
                {/* Diaphragm */}
                <path d="M35 118 Q65 108 95 120" stroke="#64748B" strokeWidth="3" fill="none" opacity="0.7" />
                <path d="M105 120 Q135 108 165 118" stroke="#64748B" strokeWidth="3" fill="none" opacity="0.7" />
                <defs>
                  <radialGradient id="lungGlow" cx="50%" cy="50%" r="50%">
                    <stop offset="0%" stopColor="#94A3B8" stopOpacity="0.45" />
                    <stop offset="70%" stopColor="#475569" stopOpacity="0.2" />
                    <stop offset="100%" stopColor="#0F172A" stopOpacity="0" />
                  </radialGradient>
                </defs>
              </svg>

              {/* Contrast glass highlight */}
              <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-white/5 to-white/20 pointer-events-none" />
            </div>
          </div>

          {/* Floating Model Badges */}
          <div className="flex flex-col gap-2.5">
            {/* ResNet50 Badge */}
            <div className="flex items-center gap-2.5 px-3 py-2 rounded-xl bg-white/90 backdrop-blur-md border border-slate-200/80 shadow-card">
              <div className="w-8 h-8 rounded-lg bg-blue-50 border border-blue-100 flex items-center justify-center text-blue-600 shrink-0">
                <Layers className="w-4 h-4" />
              </div>
              <div>
                <div className="text-xs font-bold text-slate-800 leading-tight">ResNet50</div>
                <div className="text-[10px] text-slate-500 font-medium">(Transfer Learning)</div>
              </div>
            </div>

            {/* EfficientNetB0 Badge */}
            <div className="flex items-center gap-2.5 px-3 py-2 rounded-xl bg-white/90 backdrop-blur-md border border-slate-200/80 shadow-card">
              <div className="w-8 h-8 rounded-lg bg-purple-50 border border-purple-100 flex items-center justify-center text-purple-600 shrink-0">
                <Cpu className="w-4 h-4" />
              </div>
              <div>
                <div className="text-xs font-bold text-slate-800 leading-tight">EfficientNetB0</div>
                <div className="text-[10px] text-slate-500 font-medium">(Transfer Learning)</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
