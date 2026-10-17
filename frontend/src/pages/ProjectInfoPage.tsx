import React, { useEffect, useState } from 'react';
import {
  Info,
  ShieldAlert,
  Award,
  Cpu,
  Activity,
  CheckCircle2,
  XCircle,
  Database,
  Layers,
  Flame,
  FileCode,
  Server,
  RefreshCw,
} from 'lucide-react';
import { medicalApi } from '../services/api';
import type { BackendHealth, AnalyticsSummaryResponse, ModelInfo } from '../types';

export const ProjectInfoPage: React.FC = () => {
  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [summary, setSummary] = useState<AnalyticsSummaryResponse | null>(null);
  const [projectMeta, setProjectMeta] = useState<any>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  const loadStatus = async () => {
    setLoading(true);
    try {
      const [hData, sData, pData, mData] = await Promise.all([
        medicalApi.checkHealth().catch(() => null),
        medicalApi.getAnalyticsSummary().catch(() => null),
        medicalApi.getProjectInfo().catch(() => null),
        medicalApi.getModels().catch(() => []),
      ]);
      setHealth(hData);
      setSummary(sData);
      setProjectMeta(pData);
      setModels(mData || []);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadStatus();
  }, []);

  const isConnected = Boolean(health?.isConnected);
  const pytorchVer = health?.pyTorchVersion || 'Not available';
  const anyTrained = Boolean(summary?.models?.some((m) => m.training_status === 'completed'));
  const anyEvaluated = Boolean(summary?.models?.some((m) => m.evaluation_status === 'completed'));

  const resnetModel = models.find((m) => m.name.toLowerCase().includes('resnet') || m.name.toLowerCase() === 'resnet50');
  const resnetParams = resnetModel?.total_parameters != null 
    ? `${resnetModel.total_parameters.toLocaleString()} parameters` 
    : 'Parameters not available';

  const effnetModel = models.find((m) => m.name.toLowerCase().includes('efficient') || m.name.toLowerCase() === 'efficientnet_b0');
  const effnetParams = effnetModel?.total_parameters != null 
    ? `${effnetModel.total_parameters.toLocaleString()} parameters` 
    : 'Parameters not available';

  return (
    <div className="space-y-6 animate-in fade-in duration-200">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight flex items-center gap-2.5">
            <Info className="w-6 h-6 text-blue-600" />
            Project Information & System Status
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Academic foundation, architecture specifications, real-time environment status, and safety disclaimers
          </p>
        </div>

        <button
          onClick={loadStatus}
          disabled={loading}
          className="inline-flex items-center gap-2 px-3.5 py-1.5 text-xs font-semibold text-slate-700 bg-white border border-slate-200/80 rounded-xl hover:bg-slate-50 transition-colors shadow-2xs self-start sm:self-auto"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-blue-600' : ''}`} />
          <span>Refresh Status</span>
        </button>
      </div>

      {/* -------------------------------- SECTION: REAL-TIME SYSTEM STATUS (Requirement 22) -------------------------------- */}
      <div className="bg-white rounded-2xl p-4 sm:p-6 border border-slate-200/90 shadow-card">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between mb-4 pb-2 border-b border-slate-100 gap-1.5">
          <div className="flex items-center gap-2">
            <Activity className="w-4 h-4 text-blue-600" />
            <h2 className="text-xs sm:text-sm font-bold uppercase tracking-wider text-slate-800">
              Operational System Status & Runtime Telemetry
            </h2>
          </div>
          <span className="text-[10px] sm:text-[11px] font-mono text-slate-400">Environment: {isConnected ? 'Backend Active' : 'Offline'}</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2.5 sm:gap-3">
          {/* 1. API Status */}
          <div className="p-3 sm:p-3.5 rounded-xl bg-slate-50/80 border border-slate-200/70">
            <div className="flex items-center gap-1.5 mb-1">
              <div className={`w-2 h-2 rounded-full shrink-0 ${isConnected ? 'bg-emerald-500 animate-pulse' : 'bg-rose-500'}`} />
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 truncate">FastAPI Server</span>
            </div>
            <span className="text-[11px] sm:text-xs font-bold text-slate-900 block font-mono truncate">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
            <span className="text-[10px] text-slate-400 block truncate">REST API &bull; {isConnected ? 'Active' : 'Offline'}</span>
          </div>

          {/* 2. Device */}
          <div className="p-3 sm:p-3.5 rounded-xl bg-slate-50/80 border border-slate-200/70">
            <div className="flex items-center gap-1.5 mb-1">
              <Cpu className="w-3.5 h-3.5 text-blue-600 shrink-0" />
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 truncate">Hardware</span>
            </div>
            <span className="text-[11px] sm:text-xs font-bold text-slate-900 block font-mono truncate">
              CPU (x86_64)
            </span>
            <span className="text-[10px] text-slate-400 block truncate">CUDA unavailable</span>
          </div>

          {/* 3. PyTorch */}
          <div className="p-3 sm:p-3.5 rounded-xl bg-slate-50/80 border border-slate-200/70">
            <div className="flex items-center gap-1.5 mb-1">
              <Server className="w-3.5 h-3.5 text-indigo-600 shrink-0" />
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 truncate">PyTorch</span>
            </div>
            <span className="text-[11px] sm:text-xs font-bold text-slate-900 block font-mono truncate">
              {pytorchVer}
            </span>
            <span className="text-[10px] text-slate-400 block truncate">
              Torchvision: {health?.torchvisionVersion || 'Not available'}
            </span>
          </div>

          {/* 4. Model Weights */}
          <div className="p-3 sm:p-3.5 rounded-xl bg-slate-50/80 border border-slate-200/70">
            <div className="flex items-center gap-1.5 mb-1">
              {anyTrained ? (
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
              ) : (
                <XCircle className="w-3.5 h-3.5 text-slate-400 shrink-0" />
              )}
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 truncate">Weights</span>
            </div>
            <span className="text-[11px] sm:text-xs font-bold text-slate-900 block font-mono truncate">
              {anyTrained ? 'Trained (On Disk)' : 'Not Trained'}
            </span>
            <span className="text-[10px] text-slate-400 block truncate">checkpoints/*.pt</span>
          </div>

          {/* 5. Test Evaluation */}
          <div className="p-3 sm:p-3.5 rounded-xl bg-slate-50/80 border border-slate-200/70 col-span-2 sm:col-span-1">
            <div className="flex items-center gap-1.5 mb-1">
              {anyEvaluated ? (
                <Award className="w-3.5 h-3.5 text-amber-500 shrink-0" />
              ) : (
                <XCircle className="w-3.5 h-3.5 text-slate-400 shrink-0" />
              )}
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-500 truncate">Held-Out Test</span>
            </div>
            <span className="text-[11px] sm:text-xs font-bold text-slate-900 block font-mono truncate">
              {anyEvaluated ? (projectMeta?.dataset?.splits?.test ? `Evaluated (N=${projectMeta.dataset.splits.test})` : 'Evaluated') : 'Not Evaluated'}
            </span>
            <span className="text-[10px] text-slate-400 block truncate">ROC & PR curves</span>
          </div>
        </div>
      </div>

      {/* -------------------------------- PRIMARY RESEARCH OVERVIEW -------------------------------- */}
      <div className="bg-white rounded-2xl p-4 sm:p-6 lg:p-8 border border-slate-200/90 shadow-card">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 text-blue-700 text-xs font-semibold mb-4 border border-blue-200/60">
          <Award className="w-3.5 h-3.5 text-blue-600" />
          <span>{projectMeta?.short_name || 'Academic Deep Learning Platform'}</span>
        </div>

        <h2 className="text-xl sm:text-2xl font-extrabold text-slate-900 mb-3 leading-tight">
          {projectMeta?.title ? (
            <span>{projectMeta.title}</span>
          ) : (
            <>
              Medical Image Classification Using Transfer Learning:<br className="hidden sm:inline" />
              <span className="text-blue-600"> A Comparative Study of ResNet50 and EfficientNetB0 on PneumoniaMNIST</span>
            </>
          )}
        </h2>

        <p className="text-sm text-slate-600 leading-relaxed max-w-4xl mb-6">
          {projectMeta?.purpose ? `${projectMeta.purpose}: ` : ''}This research investigation explores transfer learning mechanisms for computer-aided chest radiograph screening.
          By leveraging architectures pre-trained on ImageNet, we analyze the comparative trade-offs between deep residual representations
          (ResNet50) and compound-scaled mobile inverted bottleneck architectures (EfficientNetB0) when fine-tuned on the standardized
          PneumoniaMNIST benchmark.
        </p>

        {/* Specification Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pt-6 border-t border-slate-100">
          {/* Dataset */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/60 space-y-1">
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-blue-600" />
              Benchmark Dataset
            </span>
            <span className="text-sm font-bold text-slate-900 block">{projectMeta?.dataset?.name || 'PneumoniaMNIST (MedMNIST v2)'}</span>
            <span className="text-xs text-slate-500 block">
              {projectMeta?.dataset?.total_samples
                ? `${projectMeta.dataset.total_samples.toLocaleString()} pediatric chest radiographs (Train: ${projectMeta.dataset.splits?.train?.toLocaleString() ?? '--'} / Val: ${(projectMeta.dataset.splits?.validation ?? projectMeta.dataset.splits?.val)?.toLocaleString() ?? '--'} / Test: ${projectMeta.dataset.splits?.test?.toLocaleString() ?? '--'})`
                : 'PneumoniaMNIST partitions loading...'}
            </span>
          </div>

          {/* Classes */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/60 space-y-1">
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-indigo-600" />
              Target Classes
            </span>
            <span className="text-sm font-bold text-slate-900 block">
              Binary: {projectMeta?.dataset?.classes ? projectMeta.dataset.classes.join(' & ') : 'Normal (0) & Pneumonia (1)'}
            </span>
            <span className="text-xs text-slate-500 block">
              Supervised pediatric radiograph discrimination with CrossEntropy loss
            </span>
          </div>

          {/* Tech Stack */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/60 space-y-1">
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider flex items-center gap-1.5">
              <FileCode className="w-3.5 h-3.5 text-cyan-600" />
              Technology Stack
            </span>
            <span className="text-sm font-bold text-slate-900 block">PyTorch + FastAPI + React + Vite</span>
            <span className="text-xs text-slate-500 block">TypeScript &bull; Tailwind CSS &bull; Recharts &bull; ReportLab</span>
          </div>

          {/* ResNet50 */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/60 space-y-1">
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-blue-600" />
              ResNet50 Architecture
            </span>
            <span className="text-sm font-bold text-slate-900 block">Deep Residual Network</span>
            <span className="text-xs text-slate-500 block">{resnetParams} &bull; Bottleneck residual skip blocks &bull; layer4 target</span>
          </div>

          {/* EfficientNetB0 */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/60 space-y-1">
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-cyan-600" />
              EfficientNetB0 Architecture
            </span>
            <span className="text-sm font-bold text-slate-900 block">Compound Scaling Network</span>
            <span className="text-xs text-slate-500 block">{effnetParams} &bull; MBConv + Squeeze-and-Excitation &bull; features[-1] target</span>
          </div>

          {/* Explainability */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/60 space-y-1">
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider flex items-center gap-1.5">
              <Flame className="w-3.5 h-3.5 text-amber-500" />
              Explainability Method
            </span>
            <span className="text-sm font-bold text-slate-900 block">Grad-CAM (Gradient Class Activation)</span>
            <span className="text-xs text-slate-500 block">Convolutional feature map gradients mapped to Jet colormap heatmaps</span>
          </div>

          {/* Training Approach */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/60 space-y-1">
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider flex items-center gap-1.5">
              <Activity className="w-3.5 h-3.5 text-emerald-600" />
              Training Pipeline
            </span>
            <span className="text-sm font-bold text-slate-900 block">Transfer Learning Fine-Tuning</span>
            <span className="text-xs text-slate-500 block">AdamW Optimizer &bull; Learning rate 0.001 &bull; Cross-Entropy loss &bull; 5 Epochs</span>
          </div>

          {/* Evaluation Metrics */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/60 space-y-1">
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider flex items-center gap-1.5">
              <Award className="w-3.5 h-3.5 text-blue-600" />
              Diagnostic Evaluation Metrics
            </span>
            <span className="text-sm font-bold text-slate-900 block">7 Verified Metrics</span>
            <span className="text-xs text-slate-500 block">Accuracy, Precision, Recall, Sensitivity, Specificity, F1, ROC-AUC</span>
          </div>

          {/* Primary Metric */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/60 space-y-1">
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider flex items-center gap-1.5">
              <Award className="w-3.5 h-3.5 text-amber-500" />
              Primary Ranking Metric
            </span>
            <span className="text-sm font-bold text-slate-900 block">ROC-AUC</span>
            <span className="text-xs text-slate-500 block">Evaluates discrimination ranking across all operating decision thresholds</span>
          </div>
        </div>
      </div>

      {/* -------------------------------- RESEARCH & SAFETY DISCLAIMER -------------------------------- */}
      <div className="p-4 sm:p-6 rounded-2xl bg-amber-50/90 border border-amber-200/90 shadow-card">
        <div className="flex flex-col sm:flex-row items-start gap-3 sm:gap-4">
          <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-amber-100 flex items-center justify-center text-amber-700 shrink-0">
            <ShieldAlert className="w-5 h-5 sm:w-6 sm:h-6" />
          </div>
          <div className="space-y-1.5">
            <h3 className="text-xs sm:text-sm font-bold text-amber-950">
              Safety, Privacy, and Research Disclaimer (Mandatory)
            </h3>
            <p className="text-xs text-amber-900 leading-relaxed">
              <strong>Educational & Research Use Only.</strong> This application is not a clinically validated diagnostic system.
              Model outputs are experimental research results and should not be used as a substitute for professional medical judgment.
              Performance metrics are specific strictly to the evaluated PneumoniaMNIST research dataset and should not be interpreted as
              clinical diagnostic performance.
            </p>
            <p className="text-[11px] text-amber-800 leading-relaxed pt-1 border-t border-amber-200/60">
              <strong>Privacy Assurance:</strong> Uploaded medical radiographs are processed strictly in memory on CPU for inference
              and attention visualization. Images are never permanently retained, logged, or shared.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
