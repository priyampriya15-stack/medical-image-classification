import React from 'react';

type StatusType = 
  | 'not_started' 
  | 'queued' 
  | 'training' 
  | 'completed' 
  | 'failed'
  | 'connected'
  | 'disconnected'
  | 'normal'
  | 'pneumonia'
  | 'active'
  | 'inactive';

interface StatusBadgeProps {
  status: StatusType | string;
  label?: string;
  size?: 'sm' | 'md';
  showDot?: boolean;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  label,
  size = 'md',
  showDot = true,
}) => {
  const normalized = status.toLowerCase();

  let styles = 'bg-slate-100 text-slate-700 border-slate-200';
  let dotColor = 'bg-slate-400';
  let displayLabel = label || status;

  switch (normalized) {
    case 'completed':
    case 'connected':
    case 'normal':
    case 'active':
    case 'success':
      styles = 'bg-emerald-50 text-emerald-700 border-emerald-200/80';
      dotColor = 'bg-emerald-500';
      if (!label && normalized === 'completed') displayLabel = 'Completed';
      if (!label && normalized === 'connected') displayLabel = 'Backend Connected';
      break;

    case 'training':
    case 'running':
      styles = 'bg-blue-50 text-blue-700 border-blue-200/80 animate-pulse';
      dotColor = 'bg-blue-500';
      if (!label) displayLabel = 'Training in Progress';
      break;

    case 'queued':
      styles = 'bg-amber-50 text-amber-700 border-amber-200/80';
      dotColor = 'bg-amber-500';
      if (!label) displayLabel = 'Queued';
      break;

    case 'stopped':
      styles = 'bg-amber-50 text-amber-800 border-amber-200/80';
      dotColor = 'bg-amber-500';
      if (!label) displayLabel = 'Stopped';
      break;


    case 'not_started':
    case 'not started':
    case 'idle':
      styles = 'bg-slate-100 text-slate-600 border-slate-200';
      dotColor = 'bg-slate-400';
      if (!label) displayLabel = 'Not Started';
      break;

    case 'failed':
    case 'disconnected':
    case 'error':
      styles = 'bg-rose-50 text-rose-700 border-rose-200/80';
      dotColor = 'bg-rose-500';
      if (!label && normalized === 'disconnected') displayLabel = 'Backend Offline';
      if (!label && normalized === 'failed') displayLabel = 'Failed';
      break;

    case 'pneumonia':
      styles = 'bg-amber-50 text-amber-800 border-amber-200';
      dotColor = 'bg-amber-600';
      if (!label) displayLabel = 'Pneumonia';
      break;

    default:
      styles = 'bg-slate-100 text-slate-700 border-slate-200';
      dotColor = 'bg-slate-400';
      break;
  }

  const sizeClasses = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-xs font-medium';

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border ${styles} ${sizeClasses} transition-all`}
    >
      {showDot && (
        <span
          className={`w-1.5 h-1.5 rounded-full ${dotColor} ${
            normalized === 'training' || normalized === 'connected' ? 'animate-pulse' : ''
          }`}
        />
      )}
      <span>{displayLabel}</span>
    </span>
  );
};
