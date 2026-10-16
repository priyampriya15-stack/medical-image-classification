import React from 'react';
import {
  Activity,
  Cpu,
  Clock,
  AlertCircle,
  FileCheck2,
} from 'lucide-react';
import type { SinglePredictionResponse } from '../../types';

interface PredictionResultCardProps {
  prediction: SinglePredictionResponse;
}

export const PredictionResultCard: React.FC<PredictionResultCardProps> = ({
  prediction,
}) => {
  const predClass = prediction.predicted_class || prediction.prediction?.class || 'Normal';
  const predConf = prediction.confidence ?? prediction.prediction?.confidence ?? 0;
  const classIdx = prediction.class_index ?? prediction.prediction?.class_index ?? (predClass === 'Pneumonia' ? 1 : 0);
  const isPneumonia = predClass === 'Pneumonia';
  const confidencePercent = (predConf * 100).toFixed(2);
  const normalPercent = ((prediction.probabilities?.Normal ?? (isPneumonia ? 1 - predConf : predConf)) * 100).toFixed(2);
  const pneumoniaPercent = ((prediction.probabilities?.Pneumonia ?? (isPneumonia ? predConf : 1 - predConf)) * 100).toFixed(2);
  const modelName = prediction.model?.display_name || (prediction.model_name === 'efficientnet_b0' ? 'EfficientNetB0' : 'ResNet50');

  return (
    <div className="bg-white rounded-2xl p-4 sm:p-6 border border-slate-200/90 shadow-card space-y-5 sm:space-y-6 animate-in fade-in duration-300">
      {/* Header with Model Badge */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between border-b border-slate-100 pb-4 gap-2">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs uppercase font-bold tracking-wider text-slate-400">
              Inference Evaluation
            </span>
            <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-blue-50 text-blue-700 border border-blue-200">
              <Activity className="w-3 h-3" />
              {modelName}
            </span>
          </div>
          <h3 className="text-base sm:text-lg font-extrabold text-slate-900 mt-1">
            Model Prediction Result
          </h3>
        </div>

        <div className="self-start sm:self-auto">
          <span className="inline-flex items-center px-2 py-0.5 rounded text-[10px] font-medium bg-slate-100 text-slate-600 font-mono">
            Experimental model output
          </span>
        </div>
      </div>

      {/* Primary Classification Callout */}
      <div className="p-4 sm:p-5 rounded-2xl bg-gradient-to-br from-slate-900 to-slate-800 text-white shadow-md relative overflow-hidden">
        <div className="absolute top-0 right-0 w-36 h-36 bg-blue-500/10 rounded-full blur-2xl pointer-events-none" />

        <div className="relative z-10 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider block mb-1">
              Predicted Binary Class
            </span>
            <div className="flex flex-wrap items-center gap-2.5 sm:gap-3">
              <span className="text-xl sm:text-2xl md:text-3xl font-black tracking-tight text-white">
                {predClass}
              </span>
              <span
                className={`px-2.5 py-0.5 sm:py-1 rounded-full text-xs font-semibold border ${
                  isPneumonia
                    ? 'bg-amber-500/20 text-amber-300 border-amber-500/30'
                    : 'bg-teal-500/20 text-teal-300 border-teal-500/30'
                }`}
              >
                Index: {classIdx}
              </span>
            </div>
          </div>

          <div className="sm:text-right border-t sm:border-t-0 border-slate-700 pt-3 sm:pt-0">
            <span className="text-xs font-medium text-slate-400 uppercase tracking-wider block mb-0.5">
              Confidence Score
            </span>
            <span className="text-xl sm:text-2xl md:text-3xl font-mono font-extrabold text-blue-400 block">
              {confidencePercent}%
            </span>
            <span className="text-[11px] text-slate-400 block font-mono">
              p = {predConf.toFixed(4)}
            </span>
          </div>
        </div>
      </div>

      {/* Full Softmax Probability Distribution */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h4 className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center gap-2">
            <FileCheck2 className="w-4 h-4 text-slate-500" />
            Softmax Class Probabilities
          </h4>
          <span className="text-[11px] font-mono text-slate-400">
            sum = {(Number(normalPercent) / 100 + Number(pneumoniaPercent) / 100).toFixed(4)}
          </span>
        </div>

        <div className="space-y-4 bg-slate-50/80 p-4 rounded-xl border border-slate-200/70">
          {/* Normal Probability */}
          <div>
            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="font-semibold text-slate-800 flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-teal-500 inline-block" />
                Normal Radiograph
              </span>
              <div className="flex items-center gap-2 font-mono">
                <span className="text-slate-500 text-[11px]">
                  {(Number(normalPercent) / 100).toFixed(4)}
                </span>
                <span className="font-bold text-slate-800 text-xs">
                  {normalPercent}%
                </span>
              </div>
            </div>
            <div className="w-full h-2.5 rounded-full bg-slate-200 overflow-hidden">
              <div
                className="h-full bg-teal-600 rounded-full transition-all duration-500"
                style={{ width: `${Math.max(Number(normalPercent), 1)}%` }}
              />
            </div>
          </div>

          {/* Pneumonia Probability */}
          <div>
            <div className="flex items-center justify-between text-xs mb-1.5">
              <span className="font-semibold text-slate-800 flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-amber-500 inline-block" />
                Pneumonia Manifestation
              </span>
              <div className="flex items-center gap-2 font-mono">
                <span className="text-slate-500 text-[11px]">
                  {(Number(pneumoniaPercent) / 100).toFixed(4)}
                </span>
                <span className="font-bold text-slate-800 text-xs">
                  {pneumoniaPercent}%
                </span>
              </div>
            </div>
            <div className="w-full h-2.5 rounded-full bg-slate-200 overflow-hidden">
              <div
                className="h-full bg-amber-600 rounded-full transition-all duration-500"
                style={{ width: `${Math.max(Number(pneumoniaPercent), 1)}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Metadata & Technical Telemetry */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 pt-3 border-t border-slate-100 text-xs">
        <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/70">
          <span className="text-[10px] uppercase font-bold text-slate-400 block mb-0.5">
            Architecture
          </span>
          <span className="font-semibold text-slate-800 truncate block">
            {modelName}
          </span>
        </div>

        <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/70">
          <span className="text-[10px] uppercase font-bold text-slate-400 block mb-0.5 flex items-center gap-1">
            <Clock className="w-3 h-3 text-slate-400" />
            Inference Time
          </span>
          <span className="font-mono font-bold text-slate-800 block">
            {(prediction.inference_time_ms ?? 0).toFixed(1)} ms
          </span>
        </div>

        <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/70">
          <span className="text-[10px] uppercase font-bold text-slate-400 block mb-0.5 flex items-center gap-1">
            <Cpu className="w-3 h-3 text-slate-400" />
            Compute Device
          </span>
          <span className="font-mono font-semibold text-slate-800 uppercase block">
            {prediction.device || 'CPU'}
          </span>
        </div>

        <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/70">
          <span className="text-[10px] uppercase font-bold text-slate-400 block mb-0.5">
            Image Input
          </span>
          <span className="font-mono text-slate-700 text-[11px] truncate block">
            {prediction.image ? `${prediction.image.width}×${prediction.image.height} (${prediction.image.format})` : '224×224 (Pillow)'}
          </span>
        </div>
      </div>

      {/* Research Disclaimer */}
      <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-600 flex items-start gap-3">
        <AlertCircle className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="font-bold text-slate-800">
            Educational & Research Use Only
          </p>
          <p className="text-slate-500 leading-relaxed text-[11px]">
            This model is not clinically validated and should not be used for medical diagnosis.
            Inference outputs are experimental algorithmic probabilities intended solely for academic transfer learning research.
          </p>
        </div>
      </div>
    </div>
  );
};
