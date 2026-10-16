import React from 'react';
import { useNavigate } from 'react-router-dom';
import { FileText, ArrowRight } from 'lucide-react';

interface ProjectInfoCardProps {
  datasetName?: string;
  classCount?: number;
  modelNames?: string[];
  frameworkStack?: string;
}

export const ProjectInfoCard: React.FC<ProjectInfoCardProps> = ({
  datasetName = 'PneumoniaMNIST',
  classCount = 2,
  modelNames = ['ResNet50', 'EfficientNetB0'],
  frameworkStack = 'PyTorch + FastAPI + React',
}) => {
  const navigate = useNavigate();

  return (
    <div className="bg-white rounded-2xl p-5 border border-slate-200/80 shadow-xs">
      {/* Header */}
      <div className="flex items-center gap-2 mb-4 pb-3 border-b border-slate-100">
        <div className="w-8 h-8 rounded-lg bg-blue-50 border border-blue-100 flex items-center justify-center text-blue-600 shrink-0">
          <FileText className="w-4 h-4" />
        </div>
        <h3 className="text-sm font-bold text-slate-900 tracking-tight">
          Project Information
        </h3>
      </div>

      {/* Metadata Key-Value Rows matching screenshot */}
      <div className="space-y-3 mb-5 text-xs">
        <div className="flex items-center justify-between">
          <span className="text-slate-500 font-medium">Dataset</span>
          <span className="font-semibold text-slate-900">{datasetName}</span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-slate-500 font-medium">Classes</span>
          <span className="font-semibold text-slate-900">
            {classCount} (Normal / Pneumonia)
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-slate-500 font-medium">Models</span>
          <span className="font-semibold text-slate-900">
            {modelNames.join(', ')}
          </span>
        </div>

        <div className="flex items-center justify-between">
          <span className="text-slate-500 font-medium">Framework</span>
          <span className="font-semibold text-slate-900">{frameworkStack}</span>
        </div>
      </div>

      {/* Learn More Button */}
      <button
        onClick={() => navigate('/project-info')}
        className="w-full py-2.5 px-4 rounded-xl border border-slate-200 hover:border-blue-300 bg-white hover:bg-slate-50 text-slate-700 hover:text-blue-600 text-xs font-semibold transition-all flex items-center justify-center gap-2 cursor-pointer shadow-2xs"
      >
        <span>Learn More</span>
        <ArrowRight className="w-3.5 h-3.5" />
      </button>
    </div>
  );
};
