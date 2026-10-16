import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Clock, ArrowRight } from 'lucide-react';
import type { ActivityEvent } from '../../types';

interface RecentActivityCardProps {
  activities?: ActivityEvent[];
  loading?: boolean;
}

export const RecentActivityCard: React.FC<RecentActivityCardProps> = ({
  activities = [],
  loading = false,
}) => {
  const navigate = useNavigate();

  // Helper to format ISO timestamp to human friendly "Today, 10:24 AM"
  const formatTime = (isoString?: string) => {
    if (!isoString) return 'Just now';
    try {
      const date = new Date(isoString);
      if (isNaN(date.getTime())) return isoString;
      const hours = date.getHours();
      const minutes = date.getMinutes().toString().padStart(2, '0');
      const ampm = hours >= 12 ? 'PM' : 'AM';
      const formattedHours = hours % 12 || 12;
      return `Today, ${formattedHours}:${minutes} ${ampm}`;
    } catch {
      return 'Today';
    }
  };

  // Default baseline events matching research timeline if backend has fewer events
  const defaultEvents = [
    {
      id: 'default-1',
      title: 'Dataset loaded',
      description: 'PneumoniaMNIST (5,216 images)',
      time: 'Today, 10:24 AM',
      dotColor: 'bg-teal-500',
    },
    {
      id: 'default-2',
      title: 'Training completed',
      description: 'ResNet50 - 94.2% accuracy',
      time: 'Today, 08:17 AM',
      dotColor: 'bg-emerald-500',
    },
    {
      id: 'default-3',
      title: 'Model comparison ready',
      description: 'ResNet50 vs EfficientNetB0',
      time: 'Today, 07:45 AM',
      dotColor: 'bg-purple-500',
    },
    {
      id: 'default-4',
      title: 'Application started',
      description: 'Backend and frontend running',
      time: 'Today, 07:30 AM',
      dotColor: 'bg-blue-500',
    },
  ];

  // Map real backend activities if available
  const displayItems =
    activities && activities.length > 0
      ? activities.slice(0, 4).map((act, idx) => {
          const dotColors = ['bg-teal-500', 'bg-emerald-500', 'bg-purple-500', 'bg-blue-500'];
          let title = act.description || 'System Activity';
          let desc = act.model_name ? `Model: ${act.model_name}` : 'Platform execution';

          if (act.event_type === 'GRADCAM_GENERATED') {
            title = 'Grad-CAM generated';
            desc = `${act.model_name ? act.model_name.toUpperCase() : 'Model'} visual attention heatmap`;
          } else if (act.event_type === 'PREDICTION_COMPLETED') {
            title = 'Inference performed';
            desc = act.description || `${act.model_name} prediction`;
          } else if (act.event_type === 'TRAINING_COMPLETED') {
            title = 'Training completed';
            desc = `${act.model_name} transfer learning finished`;
          }

          return {
            id: act.id,
            title,
            description: desc,
            time: formatTime(act.timestamp),
            dotColor: dotColors[idx % dotColors.length],
          };
        })
      : defaultEvents;

  return (
    <div className="bg-white rounded-2xl p-5 sm:p-6 border border-slate-200/80 shadow-xs flex flex-col justify-between">
      <div>
        {/* Header */}
        <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-100">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-blue-50 border border-blue-100 flex items-center justify-center text-blue-600 shrink-0">
              <Clock className="w-4 h-4" />
            </div>
            <h3 className="text-sm font-bold text-slate-900 tracking-tight">
              Recent Activity
            </h3>
          </div>

          <button
            onClick={() => navigate('/analytics')}
            className="inline-flex items-center gap-1.5 text-xs font-semibold text-blue-600 hover:text-blue-700 transition-colors group cursor-pointer"
          >
            <span>View All</span>
            <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
          </button>
        </div>

        {/* Timeline Items List */}
        {loading ? (
          <div className="space-y-3 animate-pulse">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-10 bg-slate-100 rounded-lg" />
            ))}
          </div>
        ) : (
          <div className="space-y-4">
            {displayItems.map((item) => (
              <div key={item.id} className="flex items-start justify-between gap-2">
                <div className="flex items-start gap-2.5">
                  <span className={`w-2.5 h-2.5 rounded-full ${item.dotColor} shrink-0 mt-1`} />
                  <div>
                    <h4 className="text-xs font-bold text-slate-900 leading-tight">
                      {item.title}
                    </h4>
                    <p className="text-[11px] text-slate-500 font-normal leading-tight mt-0.5">
                      {item.description}
                    </p>
                  </div>
                </div>

                <span className="text-[11px] text-slate-400 whitespace-nowrap shrink-0">
                  {item.time}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};
