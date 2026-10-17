import React, { useEffect, useState } from 'react';
import {
  GitCompare,
  AlertCircle,
  RefreshCw,
  Award,
  ArrowRight,
  TrendingUp,
  Info,
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { medicalApi } from '../services/api';
import type { ModelMetadata, ModelEvaluation, ModelComparisonResponse, TrainingJob, ConfusionMatrixResponse } from '../types';
import { ModelPerformanceCard } from '../components/dashboard/ModelPerformanceCard';

export const ModelComparisonPage: React.FC = () => {
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [evaluations, setEvaluations] = useState<ModelEvaluation[]>([]);
  const [comparison, setComparison] = useState<ModelComparisonResponse | null>(null);
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([]);
  const [confusionMatrix, setConfusionMatrix] = useState<ConfusionMatrixResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [modelList, evalList, compData, jobs, cMat] = await Promise.all([
        medicalApi.getModels(),
        medicalApi.getModelEvaluations(),
        medicalApi.getEvaluationComparison().catch(() => null),
        medicalApi.getTrainingStatus().catch(() => []),
        medicalApi.getConfusionMatrix().catch(() => null),
      ]);
      setModels(modelList);
      setEvaluations(evalList);
      setComparison(compData);
      setTrainingJobs(jobs);
      setConfusionMatrix(cMat);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to connect to backend model registry');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const resnetEval = comparison?.models.find((m) => m.model_name === 'resnet50');
  const effnetEval = comparison?.models.find((m) => m.model_name === 'efficientnet_b0');

  const resnetMeta = models.find((m) => m.name === 'resnet50');
  const effnetMeta = models.find((m) => m.name === 'efficientnet_b0');

  const resnetJob = trainingJobs.find((j) => j.modelId === 'resnet50');
  const effnetJob = trainingJobs.find((j) => j.modelId === 'efficientnet_b0');

  const resnetCm = confusionMatrix?.resnet50;
  const effnetCm = confusionMatrix?.efficientnet_b0;

  const totalTestSamples = resnetCm
    ? (resnetCm.true_positive ?? 0) + (resnetCm.true_negative ?? 0) + (resnetCm.false_positive ?? 0) + (resnetCm.false_negative ?? 0)
    : null;
  const totalPneumoniaCases = resnetCm
    ? (resnetCm.true_positive ?? 0) + (resnetCm.false_negative ?? 0)
    : null;
  const totalNormalCases = resnetCm
    ? (resnetCm.true_negative ?? 0) + (resnetCm.false_positive ?? 0)
    : null;

  const bothEvaluated = Boolean(resnetEval?.evaluated && effnetEval?.evaluated);
  const bestModelKey = comparison?.best_model;
  const bestModelName =
    bestModelKey === 'efficientnet_b0'
      ? 'EfficientNetB0'
      : bestModelKey === 'resnet50'
      ? 'ResNet50'
      : 'Not determined';

  const bestRocVal =
    bestModelKey === 'efficientnet_b0'
      ? effnetEval?.roc_auc
      : bestModelKey === 'resnet50'
      ? resnetEval?.roc_auc
      : null;

  const accDiff = bothEvaluated && resnetEval?.accuracy != null && effnetEval?.accuracy != null
    ? (resnetEval.accuracy - effnetEval.accuracy) * 100
    : null;
  const precDiff = bothEvaluated && resnetEval?.precision != null && effnetEval?.precision != null
    ? (resnetEval.precision - effnetEval.precision) * 100
    : null;
  const sensDiff = bothEvaluated && resnetEval?.sensitivity != null && effnetEval?.sensitivity != null
    ? (resnetEval.sensitivity - effnetEval.sensitivity) * 100
    : null;
  const specDiff = bothEvaluated && resnetEval?.specificity != null && effnetEval?.specificity != null
    ? (resnetEval.specificity - effnetEval.specificity) * 100
    : null;
  const f1Diff = bothEvaluated && resnetEval?.f1 != null && effnetEval?.f1 != null
    ? (resnetEval.f1 - effnetEval.f1) * 100
    : null;
  const aucDiff = bothEvaluated && effnetEval?.roc_auc != null && resnetEval?.roc_auc != null
    ? (effnetEval.roc_auc - resnetEval.roc_auc) * 100
    : null;
  const paramRatio = resnetMeta?.total_parameters && effnetMeta?.total_parameters
    ? resnetMeta.total_parameters / effnetMeta.total_parameters
    : null;

  return (
    <div className="space-y-6 animate-in fade-in duration-200">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight flex items-center gap-2.5">
            <GitCompare className="w-6 h-6 text-blue-600" />
            Model Architecture & Test Benchmark Comparison
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Comparative Study of ResNet50 and EfficientNetB0 on held-out PneumoniaMNIST test benchmark
          </p>
        </div>

        <div className="flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-800 border border-emerald-200">
            <Award className="w-3.5 h-3.5 text-emerald-600" />
            <span>Primary Metric: ROC-AUC</span>
          </span>
          <button
            onClick={loadData}
            disabled={loading}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-600 bg-white border border-slate-200/80 rounded-xl hover:bg-slate-50 hover:text-slate-900 transition-colors shadow-2xs"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-blue-600' : ''}`} />
            <span>Reload</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-3">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* -------------------------------- BEST MODEL CARD (Section 9) -------------------------------- */}
      <div className="p-5 sm:p-6 rounded-2xl bg-gradient-to-r from-blue-900 via-slate-900 to-indigo-950 text-white shadow-card border border-blue-800/40">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-5 sm:gap-6">
          <div className="space-y-2 max-w-2xl">
            <div className="flex flex-wrap items-center gap-2">
              <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                Primary Ranking Profile
              </span>
              <span className="text-xs text-slate-300 font-mono">PneumoniaMNIST Test Split</span>
            </div>
            <h2 className="text-xl sm:text-2xl font-bold tracking-tight text-white flex items-center gap-2">
              <Award className="w-6 h-6 text-amber-400 shrink-0" />
              <span>Best Model: <span className="text-cyan-300">{bestModelName}</span></span>
            </h2>
            <div className="text-xs text-slate-300 space-y-1">
              <div>
                <strong>Primary ranking metric:</strong> <span className="text-amber-300 font-mono font-semibold">ROC-AUC</span>
                {bestRocVal ? ` (${(bestRocVal * 100).toFixed(2)}%)` : ''}
              </div>
              <p className="leading-relaxed text-slate-400">
                <strong>Reason:</strong> ROC-AUC evaluates ranking performance across classification thresholds.
                Best performance depends on the selected evaluation metric.
              </p>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 shrink-0 w-full lg:w-auto">
            <div className="bg-white/10 backdrop-blur-md px-5 py-3 rounded-xl border border-white/10 text-center w-full sm:w-auto">
              <span className="text-[10px] uppercase font-bold text-slate-300 tracking-wider block">ROC-AUC Score</span>
              <span className="text-xl sm:text-2xl font-mono font-extrabold text-cyan-300">
                {bestRocVal ? `${(bestRocVal * 100).toFixed(2)}%` : 'Not determined'}
              </span>
              <span className="text-[10px] text-slate-400 block font-mono">Held-Out Test</span>
            </div>
            <button
              onClick={() => navigate('/analytics')}
              className="px-4 py-3 rounded-xl bg-blue-600 hover:bg-blue-500 text-xs font-semibold text-white flex items-center justify-center gap-1.5 transition-all shadow-md min-h-[44px] w-full sm:w-auto"
            >
              <span>View Analytics</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Graphical Chart Visualization Card */}
      <ModelPerformanceCard evaluations={evaluations} loading={loading} />

      {/* -------------------------------- COMPLETE COMPARISON MATRIX (Section 8) -------------------------------- */}
      <div className="bg-white rounded-2xl p-6 border border-slate-200/90 shadow-card">
        <div className="mb-4 pb-3 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-blue-600" />
              Comparative Evaluation Matrix
            </h3>
            <p className="text-xs text-slate-500">
              Side-by-side empirical verification across ResNet50 and EfficientNetB0
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500 bg-slate-100 px-2 py-0.5 rounded-md">
            Held-Out Test Split Benchmark
          </span>
        </div>

        {/* Mobile View: Responsive Comparison Cards (md:hidden) */}
        <div className="md:hidden space-y-4">
          {/* Architecture Summary Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="p-4 rounded-xl bg-blue-50/50 border border-blue-200/70">
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold text-sm text-blue-900">ResNet50</span>
                <span className="text-[10px] font-mono font-semibold bg-white px-2 py-0.5 rounded text-blue-700 border border-blue-200">
                  {resnetMeta?.total_parameters ? `${(resnetMeta.total_parameters / 1e6).toFixed(1)}M Parameters` : 'Not available'}
                </span>
              </div>
              <p className="text-[11px] text-blue-800 leading-tight">
                Deep Residual Network with bottleneck skip connections.
              </p>
            </div>

            <div className="p-4 rounded-xl bg-cyan-50/50 border border-cyan-200/70">
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold text-sm text-cyan-900">EfficientNetB0</span>
                <span className="text-[10px] font-mono font-semibold bg-emerald-100 text-emerald-800 px-2 py-0.5 rounded">
                  {effnetMeta?.total_parameters ? `${paramRatio ? `${paramRatio.toFixed(1)}x Smaller ` : ''}(${(effnetMeta.total_parameters / 1e6).toFixed(1)}M)` : 'Not available'}
                </span>
              </div>
              <p className="text-[11px] text-cyan-800 leading-tight">
                Compound Scaling architecture with mobile inverted bottlenecks.
              </p>
            </div>
          </div>

          {/* Metric Comparison Cards */}
          <div className="space-y-3">
            {/* Metric 1: ROC-AUC */}
            <div className="p-4 rounded-xl bg-amber-50/40 border border-amber-200/80">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-1.5">
                  <Award className="w-4 h-4 text-amber-600 shrink-0" />
                  <span className="font-bold text-xs text-slate-900">ROC-AUC (Primary Metric)</span>
                </div>
                {aucDiff !== null && (
                  <span className="text-[10px] font-semibold bg-cyan-100 text-cyan-800 px-2 py-0.5 rounded">
                    Top: {aucDiff >= 0 ? `EfficientNetB0 (+${aucDiff.toFixed(2)}%)` : `ResNet50 (+${(-aucDiff).toFixed(2)}%)`}
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-2 font-mono text-center mb-2">
                <div className="p-2 rounded-lg bg-white border border-slate-200/70">
                  <span className="text-[10px] font-sans text-slate-500 block">ResNet50</span>
                  <span className="text-base font-bold text-slate-900">
                    {resnetEval?.roc_auc ? `${(resnetEval.roc_auc * 100).toFixed(2)}%` : '--'}
                  </span>
                </div>
                <div className="p-2 rounded-lg bg-cyan-50 border border-cyan-300">
                  <span className="text-[10px] font-sans text-cyan-800 font-semibold block">EfficientNetB0</span>
                  <span className="text-base font-bold text-cyan-900">
                    {effnetEval?.roc_auc ? `${(effnetEval.roc_auc * 100).toFixed(2)}%` : '--'}
                  </span>
                </div>
              </div>
              <p className="text-[11px] text-slate-500">EfficientNetB0 achieves superior threshold-independent discrimination.</p>
            </div>

            {/* Metric 2: Accuracy */}
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/70">
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold text-xs text-slate-900">Classification Accuracy</span>
                {accDiff !== null && (
                  <span className="text-[10px] font-semibold bg-blue-100 text-blue-800 px-2 py-0.5 rounded">
                    Top: {accDiff >= 0 ? `ResNet50 (+${accDiff.toFixed(2)}%)` : `EfficientNetB0 (+${(-accDiff).toFixed(2)}%)`}
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-2 font-mono text-center mb-2">
                <div className="p-2 rounded-lg bg-blue-50 border border-blue-300">
                  <span className="text-[10px] font-sans text-blue-800 font-semibold block">ResNet50</span>
                  <span className="text-base font-bold text-blue-900">
                    {resnetEval?.accuracy ? `${(resnetEval.accuracy * 100).toFixed(2)}%` : '--'}
                  </span>
                </div>
                <div className="p-2 rounded-lg bg-white border border-slate-200/70">
                  <span className="text-[10px] font-sans text-slate-500 block">EfficientNetB0</span>
                  <span className="text-base font-bold text-slate-900">
                    {effnetEval?.accuracy ? `${(effnetEval.accuracy * 100).toFixed(2)}%` : '--'}
                  </span>
                </div>
              </div>
              <p className="text-[11px] text-slate-500">
                {resnetCm && effnetCm && totalTestSamples
                  ? `ResNet50 correctly predicted ${(resnetCm.true_positive ?? 0) + (resnetCm.true_negative ?? 0)}/${totalTestSamples} vs ${(effnetCm.true_positive ?? 0) + (effnetCm.true_negative ?? 0)}/${totalTestSamples} for EfficientNetB0.`
                  : totalTestSamples
                  ? `Computed across the ${totalTestSamples} held-out PneumoniaMNIST test radiographs.`
                  : 'Computed across held-out test radiographs.'}
              </p>
            </div>

            {/* Metric 3: Sensitivity / Recall */}
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/70">
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold text-xs text-slate-900">Sensitivity (Recall)</span>
                {sensDiff !== null && (
                  <span className="text-[10px] font-semibold bg-blue-100 text-blue-800 px-2 py-0.5 rounded">
                    Top: {sensDiff >= 0 ? `ResNet50 (+${sensDiff.toFixed(2)}%)` : `EfficientNetB0 (+${(-sensDiff).toFixed(2)}%)`}
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-2 font-mono text-center mb-2">
                <div className="p-2 rounded-lg bg-blue-50 border border-blue-300">
                  <span className="text-[10px] font-sans text-blue-800 font-semibold block">ResNet50</span>
                  <span className="text-base font-bold text-blue-900">
                    {resnetEval?.sensitivity ? `${(resnetEval.sensitivity * 100).toFixed(2)}%` : '--'}
                  </span>
                </div>
                <div className="p-2 rounded-lg bg-white border border-slate-200/70">
                  <span className="text-[10px] font-sans text-slate-500 block">EfficientNetB0</span>
                  <span className="text-base font-bold text-slate-900">
                    {effnetEval?.sensitivity ? `${(effnetEval.sensitivity * 100).toFixed(2)}%` : '--'}
                  </span>
                </div>
              </div>
              <p className="text-[11px] text-slate-500">
                {resnetCm && effnetCm && totalPneumoniaCases
                  ? `Pneumonia detection rate (${resnetCm.true_positive}/${totalPneumoniaCases} vs ${effnetCm.true_positive}/${totalPneumoniaCases} true positives).`
                  : 'Pneumonia detection rate across true positives.'}
              </p>
            </div>

            {/* Metric 4: Specificity */}
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/70">
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold text-xs text-slate-900">Specificity</span>
                {specDiff !== null && (
                  <span className="text-[10px] font-semibold bg-blue-100 text-blue-800 px-2 py-0.5 rounded">
                    Top: {specDiff >= 0 ? `ResNet50 (+${specDiff.toFixed(2)}%)` : `EfficientNetB0 (+${(-specDiff).toFixed(2)}%)`}
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-2 font-mono text-center mb-2">
                <div className="p-2 rounded-lg bg-blue-50 border border-blue-300">
                  <span className="text-[10px] font-sans text-blue-800 font-semibold block">ResNet50</span>
                  <span className="text-base font-bold text-blue-900">
                    {resnetEval?.specificity ? `${(resnetEval.specificity * 100).toFixed(2)}%` : '--'}
                  </span>
                </div>
                <div className="p-2 rounded-lg bg-white border border-slate-200/70">
                  <span className="text-[10px] font-sans text-slate-500 block">EfficientNetB0</span>
                  <span className="text-base font-bold text-slate-900">
                    {effnetEval?.specificity ? `${(effnetEval.specificity * 100).toFixed(2)}%` : '--'}
                  </span>
                </div>
              </div>
              <p className="text-[11px] text-slate-500">
                {resnetCm && effnetCm && totalNormalCases
                  ? `Normal radiograph true negative rate (${resnetCm.true_negative}/${totalNormalCases} vs ${effnetCm.true_negative}/${totalNormalCases}).`
                  : 'Normal radiograph true negative rate.'}
              </p>
            </div>

            {/* Metric 5: F1 Score & Precision */}
            <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/70">
              <div className="flex items-center justify-between mb-2">
                <span className="font-bold text-xs text-slate-900">F1 Score & Precision</span>
                {f1Diff !== null && (
                  <span className="text-[10px] font-semibold bg-blue-100 text-blue-800 px-2 py-0.5 rounded">
                    Top: {f1Diff >= 0 ? `ResNet50 (+${f1Diff.toFixed(2)}%)` : `EfficientNetB0 (+${(-f1Diff).toFixed(2)}%)`}
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 gap-2 font-mono text-center mb-2">
                <div className="p-2 rounded-lg bg-blue-50 border border-blue-300">
                  <span className="text-[10px] font-sans text-blue-800 font-semibold block">ResNet50</span>
                  <span className="text-xs text-slate-600 block">F1: <strong className="text-blue-900">{resnetEval?.f1 ? `${(resnetEval.f1 * 100).toFixed(2)}%` : '--'}</strong></span>
                  <span className="text-[11px] text-slate-500 block">Prec: {resnetEval?.precision ? `${(resnetEval.precision * 100).toFixed(2)}%` : '--'}</span>
                </div>
                <div className="p-2 rounded-lg bg-white border border-slate-200/70">
                  <span className="text-[10px] font-sans text-slate-500 block">EfficientNetB0</span>
                  <span className="text-xs text-slate-600 block">F1: <strong className="text-slate-900">{effnetEval?.f1 ? `${(effnetEval.f1 * 100).toFixed(2)}%` : '--'}</strong></span>
                  <span className="text-[11px] text-slate-500 block">Prec: {effnetEval?.precision ? `${(effnetEval.precision * 100).toFixed(2)}%` : '--'}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Desktop / Tablet View: Full Side-by-Side Table (hidden md:block) */}
        <div className="hidden md:block overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-50 text-slate-600 uppercase font-bold tracking-wider border-b border-slate-200">
              <tr>
                <th className="py-3.5 px-4">Evaluation Dimension</th>
                <th className="py-3.5 px-4 text-blue-700">ResNet50</th>
                <th className="py-3.5 px-4 text-cyan-700">EfficientNetB0</th>
                <th className="py-3.5 px-4 text-slate-600">Empirical Synthesis</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono">
              {/* Row 1: Architecture */}
              <tr>
                <td className="py-3 px-4 font-sans font-semibold text-slate-800">Architecture</td>
                <td className="py-3 px-4 text-slate-800 font-sans">Deep Residual Network (Bottleneck)</td>
                <td className="py-3 px-4 text-slate-800 font-sans">Compound Scaling (MBConv + SE)</td>
                <td className="py-3 px-4 font-sans text-slate-500">Residual skips vs Mobile inverted bottlenecks</td>
              </tr>

              {/* Row 2: Total Parameters */}
              <tr>
                <td className="py-3 px-4 font-sans font-semibold text-slate-800">Parameters</td>
                <td className="py-3 px-4 text-slate-800">
                  {resnetMeta?.total_parameters ? `${resnetMeta.total_parameters.toLocaleString()} (${(resnetMeta.total_parameters / 1e6).toFixed(1)}M)` : 'Not available'}
                </td>
                <td className="py-3 px-4 font-bold text-emerald-700 bg-emerald-50/40">
                  {effnetMeta?.total_parameters ? `${effnetMeta.total_parameters.toLocaleString()} (${(effnetMeta.total_parameters / 1e6).toFixed(1)}M)` : 'Not available'}
                  {paramRatio && (
                    <span className="ml-2 font-sans font-semibold text-[10px] text-emerald-800 bg-emerald-100 px-1.5 py-0.2 rounded">
                      {paramRatio.toFixed(1)}x Smaller *
                    </span>
                  )}
                </td>
                <td className="py-3 px-4 font-sans text-slate-500">EfficientNetB0 drastically reduces parameter footprint</td>
              </tr>

              {/* Row 3: Trainable Parameters */}
              <tr>
                <td className="py-3 px-4 font-sans font-semibold text-slate-800">Trainable Parameters</td>
                <td className="py-3 px-4 text-slate-800">
                  {resnetMeta?.trainable_parameters ? resnetMeta.trainable_parameters.toLocaleString() : 'Not available'}
                </td>
                <td className="py-3 px-4 font-bold text-emerald-700 bg-emerald-50/40">
                  {effnetMeta?.trainable_parameters ? effnetMeta.trainable_parameters.toLocaleString() : 'Not available'}
                </td>
                <td className="py-3 px-4 font-sans text-slate-500">Full backbone fine-tuning parameters on CPU</td>
              </tr>

              {/* Row 4: Training Status */}
              <tr>
                <td className="py-3 px-4 font-sans font-semibold text-slate-800">Training Status</td>
                <td className="py-3 px-4 font-sans font-semibold text-emerald-700">
                  {resnetJob?.status === 'completed'
                    ? `Completed (${resnetJob.currentEpoch ?? resnetJob.totalEpochs} Epochs)`
                    : resnetJob?.status
                    ? resnetJob.status
                    : 'Not available'}
                </td>
                <td className="py-3 px-4 font-sans font-semibold text-emerald-700">
                  {effnetJob?.status === 'completed'
                    ? `Completed (${effnetJob.currentEpoch ?? effnetJob.totalEpochs} Epochs)`
                    : effnetJob?.status
                    ? effnetJob.status
                    : 'Not available'}
                </td>
                <td className="py-3 px-4 font-sans text-slate-500">Transfer learning checkpoints and telemetry</td>
              </tr>

              {/* Row 5: Evaluation Status */}
              <tr>
                <td className="py-3 px-4 font-sans font-semibold text-slate-800">Evaluation Status</td>
                <td className="py-3 px-4 font-sans font-semibold text-blue-700">
                  {resnetEval?.evaluated ? 'Evaluated (Test Split)' : 'Not evaluated'}
                </td>
                <td className="py-3 px-4 font-sans font-semibold text-blue-700">
                  {effnetEval?.evaluated ? 'Evaluated (Test Split)' : 'Not evaluated'}
                </td>
                <td className="py-3 px-4 font-sans text-slate-500">Held-out test set verified deterministically</td>
              </tr>

              {/* Row 6: Accuracy */}
              <tr>
                <td className="py-3 px-4 font-sans font-semibold text-slate-800">Accuracy</td>
                <td className="py-3 px-4 font-bold text-blue-700 bg-blue-50/30">
                  {resnetEval?.accuracy ? `${(resnetEval.accuracy * 100).toFixed(2)}%` : 'Not evaluated'}
                  {accDiff !== null && accDiff > 0 && (
                    <span className="ml-2 font-sans font-semibold text-[10px] text-blue-800 bg-blue-100 px-1.5 py-0.2 rounded">
                      +{accDiff.toFixed(2)}% *
                    </span>
                  )}
                </td>
                <td className="py-3 px-4 text-slate-800">
                  {effnetEval?.accuracy ? `${(effnetEval.accuracy * 100).toFixed(2)}%` : 'Not evaluated'}
                </td>
                <td className="py-3 px-4 font-sans text-slate-500">
                  {resnetCm && effnetCm && totalTestSamples
                    ? `ResNet50 correctly predicted ${(resnetCm.true_positive ?? 0) + (resnetCm.true_negative ?? 0)}/${totalTestSamples} vs ${(effnetCm.true_positive ?? 0) + (effnetCm.true_negative ?? 0)}/${totalTestSamples} for EfficientNetB0`
                    : 'Evaluated on held-out test split'}
                </td>
              </tr>

              {/* Row 7: Precision */}
              <tr>
                <td className="py-3 px-4 font-sans font-semibold text-slate-800">Precision</td>
                <td className="py-3 px-4 font-bold text-blue-700 bg-blue-50/30">
                  {resnetEval?.precision ? `${(resnetEval.precision * 100).toFixed(2)}%` : 'Not evaluated'}
                  {precDiff !== null && precDiff > 0 && (
                    <span className="ml-2 font-sans font-semibold text-[10px] text-blue-800 bg-blue-100 px-1.5 py-0.2 rounded">
                      +{precDiff.toFixed(2)}% *
                    </span>
                  )}
                </td>
                <td className="py-3 px-4 text-slate-800">
                  {effnetEval?.precision ? `${(effnetEval.precision * 100).toFixed(2)}%` : 'Not evaluated'}
                </td>
                <td className="py-3 px-4 font-sans text-slate-500">Positive predictive rate for flagged pneumonia findings</td>
              </tr>

              {/* Row 8: Sensitivity / Recall */}
              <tr>
                <td className="py-3 px-4 font-sans font-semibold text-slate-800">
                  Sensitivity <span className="text-[10px] text-slate-400 font-normal">(Recall)</span>
                </td>
                <td className="py-3 px-4 font-bold text-blue-700 bg-blue-50/30">
                  {resnetEval?.sensitivity ? `${(resnetEval.sensitivity * 100).toFixed(2)}%` : 'Not evaluated'}
                  {sensDiff !== null && sensDiff > 0 && (
                    <span className="ml-2 font-sans font-semibold text-[10px] text-blue-800 bg-blue-100 px-1.5 py-0.2 rounded">
                      +{sensDiff.toFixed(2)}% *
                    </span>
                  )}
                </td>
                <td className="py-3 px-4 text-slate-800">
                  {effnetEval?.sensitivity ? `${(effnetEval.sensitivity * 100).toFixed(2)}%` : 'Not evaluated'}
                </td>
                <td className="py-3 px-4 font-sans text-slate-500">
                  {resnetCm && effnetCm && totalPneumoniaCases
                    ? `Pneumonia detection rate (${resnetCm.true_positive}/${totalPneumoniaCases} vs ${effnetCm.true_positive}/${totalPneumoniaCases} true positives)`
                    : 'Pneumonia detection rate'}
                </td>
              </tr>

              {/* Row 9: Specificity */}
              <tr>
                <td className="py-3 px-4 font-sans font-semibold text-slate-800">Specificity</td>
                <td className="py-3 px-4 font-bold text-blue-700 bg-blue-50/30">
                  {resnetEval?.specificity ? `${(resnetEval.specificity * 100).toFixed(2)}%` : 'Not evaluated'}
                  {specDiff !== null && specDiff > 0 && (
                    <span className="ml-2 font-sans font-semibold text-[10px] text-blue-800 bg-blue-100 px-1.5 py-0.2 rounded">
                      +{specDiff.toFixed(2)}% *
                    </span>
                  )}
                </td>
                <td className="py-3 px-4 text-slate-800">
                  {effnetEval?.specificity ? `${(effnetEval.specificity * 100).toFixed(2)}%` : 'Not evaluated'}
                </td>
                <td className="py-3 px-4 font-sans text-slate-500">
                  {resnetCm && effnetCm && totalNormalCases
                    ? `Normal true negative rate (${resnetCm.true_negative}/${totalNormalCases} vs ${effnetCm.true_negative}/${totalNormalCases})`
                    : 'Normal true negative rate'}
                </td>
              </tr>

              {/* Row 10: F1 Score */}
              <tr>
                <td className="py-3 px-4 font-sans font-semibold text-slate-800">F1</td>
                <td className="py-3 px-4 font-bold text-blue-700 bg-blue-50/30">
                  {resnetEval?.f1 ? `${(resnetEval.f1 * 100).toFixed(2)}%` : 'Not evaluated'}
                  {f1Diff !== null && f1Diff > 0 && (
                    <span className="ml-2 font-sans font-semibold text-[10px] text-blue-800 bg-blue-100 px-1.5 py-0.2 rounded">
                      +{f1Diff.toFixed(2)}% *
                    </span>
                  )}
                </td>
                <td className="py-3 px-4 text-slate-800">
                  {effnetEval?.f1 ? `${(effnetEval.f1 * 100).toFixed(2)}%` : 'Not evaluated'}
                </td>
                <td className="py-3 px-4 font-sans text-slate-500">Harmonic balance between precision and sensitivity</td>
              </tr>

              {/* Row 11: ROC-AUC (Primary Metric) */}
              <tr className="bg-amber-50/20">
                <td className="py-3.5 px-4 font-sans font-bold text-slate-900 flex items-center gap-1.5">
                  <Award className="w-3.5 h-3.5 text-amber-600" />
                  ROC-AUC <span className="text-[10px] text-amber-700 font-bold">(Primary Ranking Metric)</span>
                </td>
                <td className="py-3.5 px-4 text-slate-800 font-bold">
                  {resnetEval?.roc_auc ? `${(resnetEval.roc_auc * 100).toFixed(2)}%` : 'Not evaluated'}
                </td>
                <td className="py-3.5 px-4 font-bold text-cyan-800 bg-cyan-50/50">
                  {effnetEval?.roc_auc ? `${(effnetEval.roc_auc * 100).toFixed(2)}%` : 'Not evaluated'}
                  {aucDiff !== null && aucDiff > 0 && (
                    <span className="ml-2 font-sans font-semibold text-[10px] text-cyan-900 bg-cyan-200 px-1.5 py-0.2 rounded">
                      +{aucDiff.toFixed(2)}% Top AUC *
                    </span>
                  )}
                </td>
                <td className="py-3.5 px-4 font-sans text-slate-600 font-medium">
                  EfficientNetB0 achieves superior threshold-independent discrimination
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Mandatory Research Notice */}
        <div className="p-4 mt-4 bg-slate-50 rounded-xl border border-slate-200/80 text-xs text-slate-600 space-y-1">
          <div className="font-bold text-slate-900 flex items-center gap-1.5">
            <Info className="w-3.5 h-3.5 text-blue-600" />
            Decision Context & Scientific Nuance:
          </div>
          <p className="leading-relaxed">
            {bothEvaluated && resnetEval && effnetEval ? (
              <>
                <strong>Best performance depends on the selected evaluation metric.</strong> While EfficientNetB0 achieves a
                higher ROC-AUC ({(effnetEval.roc_auc ? effnetEval.roc_auc * 100 : 0).toFixed(2)}% vs {(resnetEval.roc_auc ? resnetEval.roc_auc * 100 : 0).toFixed(2)}%)
                {paramRatio ? ` with ${paramRatio.toFixed(1)}x fewer parameters` : ''}, ResNet50 demonstrates higher fixed-threshold accuracy
                ({(resnetEval.accuracy ? resnetEval.accuracy * 100 : 0).toFixed(2)}% vs {(effnetEval.accuracy ? effnetEval.accuracy * 100 : 0).toFixed(2)}%) and higher sensitivity
                ({(resnetEval.sensitivity ? resnetEval.sensitivity * 100 : 0).toFixed(2)}% vs {(effnetEval.sensitivity ? effnetEval.sensitivity * 100 : 0).toFixed(2)}%). Neither model is universally superior across all operational regimes.
              </>
            ) : (
              'Evaluation benchmarks are actively derived from the held-out PneumoniaMNIST test partition.'
            )}
          </p>
        </div>
      </div>
    </div>
  );
};
