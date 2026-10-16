import React, { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import {
  LineChart as AnalyticsIcon,
  Award,
  RefreshCw,
  AlertCircle,
  Layers,
  FileJson,
  FileSpreadsheet,
  FileText,
  CheckCircle2,
  GitCompare,
  TrendingUp,
  Info,
  ShieldAlert,
} from 'lucide-react';

import type {
  ModelId,
  FullEvaluationReport,
  TrainingJob,
  AnalyticsSummaryResponse,
  ConfusionMatrixResponse,
  RocResponse,
  PrecisionRecallResponse,
  TrainingHistoryResponse,
} from '../types';
import { medicalApi, triggerBrowserDownload } from '../services/api';
import { EmptyState } from '../components/common/EmptyState';

export const TrainingAnalyticsPage: React.FC = () => {
  const [selectedModel, setSelectedModel] = useState<ModelId>('resnet50');
  const [resnetReport, setResnetReport] = useState<FullEvaluationReport | null>(null);
  const [effnetReport, setEffnetReport] = useState<FullEvaluationReport | null>(null);
  const [resnetJob, setResnetJob] = useState<TrainingJob | null>(null);
  const [effnetJob, setEffnetJob] = useState<TrainingJob | null>(null);
  const [summary, setSummary] = useState<AnalyticsSummaryResponse | null>(null);
  const [trainingHistory, setTrainingHistory] = useState<TrainingHistoryResponse | null>(null);
  const [confusionMatrix, setConfusionMatrix] = useState<ConfusionMatrixResponse | null>(null);
  const [rocData, setRocData] = useState<RocResponse | null>(null);
  const [precisionRecallData, setPrecisionRecallData] = useState<PrecisionRecallResponse | null>(null);

  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Export states
  const [exportingJson, setExportingJson] = useState<boolean>(false);
  const [exportingCsv, setExportingCsv] = useState<boolean>(false);
  const [exportingPdf, setExportingPdf] = useState<boolean>(false);
  const [exportSuccessMsg, setExportSuccessMsg] = useState<string | null>(null);

  const loadAllAnalytics = async () => {
    setLoading(true);
    setError(null);
    try {
      const [rRep, eRep, rJob, eJob, sum, tHist, cMat, rData, prData] = await Promise.all([
        medicalApi.getEvaluationResult('resnet50').catch(() => null),
        medicalApi.getEvaluationResult('efficientnet_b0').catch(() => null),
        medicalApi.getModelTrainingStatus('resnet50').catch(() => null),
        medicalApi.getModelTrainingStatus('efficientnet_b0').catch(() => null),
        medicalApi.getAnalyticsSummary().catch(() => null),
        medicalApi.getTrainingHistory().catch(() => null),
        medicalApi.getConfusionMatrix().catch(() => null),
        medicalApi.getRocCurve().catch(() => null),
        medicalApi.getPrecisionRecallCurve().catch(() => null),
      ]);

      setResnetReport(rRep);
      setEffnetReport(eRep);
      setResnetJob(rJob);
      setEffnetJob(eJob);
      setSummary(sum);
      setTrainingHistory(tHist);
      setConfusionMatrix(cMat);
      setRocData(rData);
      setPrecisionRecallData(prData);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to load research analytics.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAllAnalytics();
  }, []);

  // Handlers for Exports
  const handleExportJson = async () => {
    setExportingJson(true);
    setExportSuccessMsg(null);
    try {
      const blob = await medicalApi.exportJson();
      triggerBrowserDownload(blob, 'medical_image_classification_results.json');
      setExportSuccessMsg('JSON research export downloaded successfully.');
    } catch (err: unknown) {
      setError('Failed to export JSON results.');
    } finally {
      setExportingJson(false);
    }
  };

  const handleExportCsv = async () => {
    setExportingCsv(true);
    setExportSuccessMsg(null);
    try {
      const blob = await medicalApi.exportCsv();
      triggerBrowserDownload(blob, 'medical_image_classification_results.csv');
      setExportSuccessMsg('CSV metrics table downloaded successfully.');
    } catch (err: unknown) {
      setError('Failed to export CSV results.');
    } finally {
      setExportingCsv(false);
    }
  };

  const handleExportPdf = async () => {
    setExportingPdf(true);
    setExportSuccessMsg(null);
    try {
      const blob = await medicalApi.exportReportPdf();
      triggerBrowserDownload(blob, 'medical_image_classification_report.pdf');
      setExportSuccessMsg('PDF Research Report generated and downloaded successfully.');
    } catch (err: unknown) {
      setError('Failed to generate PDF research report.');
    } finally {
      setExportingPdf(false);
    }
  };

  // Active selected model details
  const activeReport = selectedModel === 'resnet50' ? resnetReport : effnetReport;
  const activeJob = selectedModel === 'resnet50' ? resnetJob : effnetJob;
  const m = activeReport?.metrics;

  // Confusion Matrices
  const resnetCm = (confusionMatrix && confusionMatrix.resnet50)
    ? confusionMatrix.resnet50
    : resnetReport?.confusion_matrix;
  const effnetCm = (confusionMatrix && confusionMatrix.efficientnet_b0)
    ? confusionMatrix.efficientnet_b0
    : effnetReport?.confusion_matrix;

  // Transform Training Convergence History for Recharts
  const rawHistory = (trainingHistory && trainingHistory[selectedModel]) || activeJob?.history || [];
  const trainingConvergenceData = rawHistory.map((item: any) => ({
    epoch: `Ep ${item.epoch}`,
    trainLoss: item.train_loss,
    valLoss: item.validation_loss ?? item.val_loss,
    trainAcc: Number(((item.train_accuracy ?? item.train_acc ?? 0) * 100).toFixed(2)),
    valAcc: Number(((item.validation_accuracy ?? item.val_acc ?? 0) * 100).toFixed(2)),
    duration: item.epoch_duration_seconds,
  }));

  // Combined ROC Curve Data across both models
  const buildCombinedRocData = () => {
    const points: { fpr: number; resnetTpr?: number; effnetTpr?: number; randomGuess: number }[] = [];
    const rFpr = rocData?.resnet50?.false_positive_rate || resnetReport?.roc_curve?.fpr || [];
    const rTpr = rocData?.resnet50?.true_positive_rate || resnetReport?.roc_curve?.tpr || [];
    const eFpr = rocData?.efficientnet_b0?.false_positive_rate || effnetReport?.roc_curve?.fpr || [];
    const eTpr = rocData?.efficientnet_b0?.true_positive_rate || effnetReport?.roc_curve?.tpr || [];

    if (rFpr.length === 0 && eFpr.length === 0) {
      return [];
    }

    // Distinct FPR values sorted
    const allFprs = Array.from(new Set([...rFpr, ...eFpr])).sort((a, b) => a - b);
    const step = Math.max(1, Math.floor(allFprs.length / 50)); // cap at ~50 points for rendering performance

    for (let i = 0; i < allFprs.length; i += step) {
      const targetFpr = allFprs[i];
      // Closest TPR in ResNet
      let bestRTpr: number | undefined = undefined;
      for (let j = 0; j < rFpr.length; j++) {
        if (rFpr[j] <= targetFpr) bestRTpr = rTpr[j];
      }
      // Closest TPR in EffNet
      let bestETpr: number | undefined = undefined;
      for (let j = 0; j < eFpr.length; j++) {
        if (eFpr[j] <= targetFpr) bestETpr = eTpr[j];
      }

      points.push({
        fpr: Number((targetFpr * 100).toFixed(1)),
        resnetTpr: bestRTpr !== undefined ? Number((bestRTpr * 100).toFixed(1)) : undefined,
        effnetTpr: bestETpr !== undefined ? Number((bestETpr * 100).toFixed(1)) : undefined,
        randomGuess: Number((targetFpr * 100).toFixed(1)),
      });
    }

    // Ensure 100% endpoint
    points.push({ fpr: 100, resnetTpr: 100, effnetTpr: 100, randomGuess: 100 });
    return points;
  };

  const combinedRocData = buildCombinedRocData();

  // Active Model PR Curve points
  const activePrRecalls = precisionRecallData?.[selectedModel]?.recall || activeReport?.precision_recall_curve?.recall || [];
  const activePrPrecisions = precisionRecallData?.[selectedModel]?.precision || activeReport?.precision_recall_curve?.precision || [];
  const activePrData = activePrRecalls.map((recVal, idx) => ({
    recall: Number((recVal * 100).toFixed(1)),
    precision: Number(((activePrPrecisions[idx] || 0) * 100).toFixed(1)),
  }));

  return (
    <div className="space-y-8 animate-in fade-in duration-200">
      {/* -------------------------------- PAGE HEADER & EXPORT ACTIONS -------------------------------- */}
      <div className="bg-white rounded-2xl p-4 sm:p-6 border border-slate-200/90 shadow-card">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-5 sm:gap-6">
          <div>
            <div className="flex flex-wrap items-center gap-2 mb-1.5">
              <span className="text-[11px] font-bold uppercase tracking-wider text-blue-600 bg-blue-50 px-2.5 py-0.5 rounded-full border border-blue-200/60">
                PneumoniaMNIST Deep Learning Benchmark
              </span>
              <span className="text-xs text-slate-400 font-mono">Held-Out Test Split {resnetReport?.num_samples ? `(N=${resnetReport.num_samples})` : ''}</span>
            </div>
            <h1 className="text-xl sm:text-2xl md:text-3xl font-extrabold text-slate-900 tracking-tight flex items-center gap-2.5">
              <AnalyticsIcon className="w-6 h-6 sm:w-7 sm:h-7 text-blue-600 shrink-0" />
              <span>Research Analytics & Performance Report</span>
            </h1>
            <p className="text-xs text-slate-500 mt-1">
              End-to-end empirical evaluation, ROC-AUC threshold analysis, confusion matrices, and academic export
            </p>
          </div>

          {/* Export Action Buttons (Milestone 5 Section 19 & 20) */}
          <div className="flex flex-wrap items-center gap-2 sm:gap-2.5 w-full lg:w-auto">
            <button
              onClick={handleExportJson}
              disabled={exportingJson || loading}
              className="flex-1 sm:flex-initial inline-flex items-center justify-center gap-2 px-3.5 py-2.5 sm:py-2 text-xs font-semibold rounded-xl border border-slate-200 text-slate-700 bg-white hover:bg-slate-50 disabled:opacity-50 transition-all shadow-2xs min-h-[40px]"
              title="Download raw structured results in JSON format"
            >
              <FileJson className={`w-4 h-4 text-emerald-600 ${exportingJson ? 'animate-spin' : ''}`} />
              <span>{exportingJson ? 'Exporting...' : 'Export JSON'}</span>
            </button>

            <button
              onClick={handleExportCsv}
              disabled={exportingCsv || loading}
              className="flex-1 sm:flex-initial inline-flex items-center justify-center gap-2 px-3.5 py-2.5 sm:py-2 text-xs font-semibold rounded-xl border border-slate-200 text-slate-700 bg-white hover:bg-slate-50 disabled:opacity-50 transition-all shadow-2xs min-h-[40px]"
              title="Download tabular metrics in CSV format"
            >
              <FileSpreadsheet className={`w-4 h-4 text-blue-600 ${exportingCsv ? 'animate-spin' : ''}`} />
              <span>{exportingCsv ? 'Exporting...' : 'Export CSV'}</span>
            </button>

            <button
              onClick={handleExportPdf}
              disabled={exportingPdf || loading}
              className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-4 py-2.5 sm:py-2 text-xs font-semibold rounded-xl text-white bg-blue-600 hover:bg-blue-500 disabled:opacity-50 transition-all shadow-sm min-h-[40px]"
              title="Generate academic PDF research report"
            >
              <FileText className={`w-4 h-4 text-white ${exportingPdf ? 'animate-spin' : ''}`} />
              <span>{exportingPdf ? 'Generating PDF...' : 'Export PDF Report'}</span>
            </button>

            <button
              onClick={loadAllAnalytics}
              disabled={loading}
              className="p-2.5 rounded-xl border border-slate-200 text-slate-500 hover:text-slate-800 hover:bg-slate-50 transition-colors shadow-2xs min-h-[40px] min-w-[40px] flex items-center justify-center shrink-0"
              title="Reload all analytics data"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-blue-600' : ''}`} />
            </button>
          </div>
        </div>

        {/* Export Feedback Banner */}
        {exportSuccessMsg && (
          <div className="mt-4 p-3 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-center justify-between animate-in fade-in">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>{exportSuccessMsg}</span>
            </div>
            <button
              onClick={() => setExportSuccessMsg(null)}
              className="text-xs text-emerald-600 hover:text-emerald-900 font-bold"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Global Error Banner */}
        {error && (
          <div className="mt-4 p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* -------------------------------- MODEL SWITCHER TOGGLE -------------------------------- */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 rounded-2xl bg-white border border-slate-200/90 shadow-card">
        <div className="flex flex-col sm:flex-row sm:items-center gap-2 w-full sm:w-auto">
          <span className="text-xs font-bold text-slate-700 uppercase tracking-wider shrink-0">
            Active Architecture View:
          </span>
          <div className="flex flex-col sm:flex-row p-1 bg-slate-100 rounded-xl gap-1 w-full sm:w-auto">
            <button
              onClick={() => setSelectedModel('resnet50')}
              className={`w-full sm:w-auto px-3.5 py-2 sm:py-1.5 rounded-lg text-xs font-semibold transition-all min-h-[36px] flex items-center justify-center ${
                selectedModel === 'resnet50'
                  ? 'bg-white text-blue-700 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              ResNet50 (23.5M Parameters)
            </button>
            <button
              onClick={() => setSelectedModel('efficientnet_b0')}
              className={`w-full sm:w-auto px-3.5 py-2 sm:py-1.5 rounded-lg text-xs font-semibold transition-all min-h-[36px] flex items-center justify-center ${
                selectedModel === 'efficientnet_b0'
                  ? 'bg-white text-cyan-700 shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              EfficientNetB0 (4.0M Parameters)
            </button>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs text-slate-500 font-mono self-start sm:self-auto">
          <span>Target: CPU</span>
          <span>&bull;</span>
          <span>AdamW (lr=0.001)</span>
        </div>
      </div>

      {/* -------------------------------- SECTION 1: TRAINING PERFORMANCE -------------------------------- */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-blue-600" />
            Section 1 &bull; Training Performance & Convergence
          </h2>
          <span className="text-xs text-slate-400 font-mono">
            {selectedModel === 'resnet50' ? 'ResNet50' : 'EfficientNetB0'} &bull; 5 Epochs Fine-Tuning
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Chart 1: Training vs Validation Loss */}
          <div className="bg-white rounded-2xl p-6 border border-slate-200/90 shadow-card">
            <div className="mb-4 pb-2 border-b border-slate-100 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-slate-900">Training vs Validation Loss</h3>
                <p className="text-[11px] text-slate-500">Cross-entropy loss minimization across training epochs</p>
              </div>
              <span className="text-xs font-mono text-slate-500 bg-slate-50 px-2 py-0.5 rounded border border-slate-200/60">
                Loss Trend
              </span>
            </div>

            {trainingConvergenceData.length > 0 ? (
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trainingConvergenceData} margin={{ top: 10, right: 15, left: -10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F1F5F9" />
                    <XAxis dataKey="epoch" tick={{ fill: '#64748B', fontSize: 11 }} />
                    <YAxis tick={{ fill: '#64748B', fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0F233A',
                        borderColor: '#1E3A5F',
                        borderRadius: '12px',
                        color: '#FFFFFF',
                        fontSize: '12px',
                      }}
                    />
                    <Legend verticalAlign="top" align="right" wrapperStyle={{ paddingBottom: '8px', fontSize: '11px' }} />
                    <Line
                      type="monotone"
                      dataKey="trainLoss"
                      name="Training Loss"
                      stroke="#F59E0B"
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="valLoss"
                      name="Validation Loss"
                      stroke="#EF4444"
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <EmptyState
                title="Training Loss Unavailable"
                description="No training convergence records found for this architecture."
                compact
              />
            )}
          </div>

          {/* Chart 2: Training vs Validation Accuracy */}
          <div className="bg-white rounded-2xl p-6 border border-slate-200/90 shadow-card">
            <div className="mb-4 pb-2 border-b border-slate-100 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-slate-900">Training vs Validation Accuracy</h3>
                <p className="text-[11px] text-slate-500">Epoch-wise classification accuracy convergence on PneumoniaMNIST</p>
              </div>
              <span className="text-xs font-mono font-bold text-emerald-700 bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
                Best: {activeJob?.bestValScore ? `${(activeJob.bestValScore * 100).toFixed(2)}%` : '--'}
              </span>
            </div>

            {trainingConvergenceData.length > 0 ? (
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={trainingConvergenceData} margin={{ top: 10, right: 15, left: -10, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F1F5F9" />
                    <XAxis dataKey="epoch" tick={{ fill: '#64748B', fontSize: 11 }} />
                    <YAxis domain={[70, 100]} unit="%" tick={{ fill: '#64748B', fontSize: 11 }} />
                    <Tooltip
                      formatter={(val: any) => [`${val}%`, 'Accuracy']}
                      contentStyle={{
                        backgroundColor: '#0F233A',
                        borderColor: '#1E3A5F',
                        borderRadius: '12px',
                        color: '#FFFFFF',
                        fontSize: '12px',
                      }}
                    />
                    <Legend verticalAlign="top" align="right" wrapperStyle={{ paddingBottom: '8px', fontSize: '11px' }} />
                    <Line
                      type="monotone"
                      dataKey="trainAcc"
                      name="Train Accuracy (%)"
                      stroke="#3B82F6"
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="valAcc"
                      name="Val Accuracy (%)"
                      stroke="#10B981"
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <EmptyState
                title="Accuracy Metrics Unavailable"
                description="No accuracy progression data available for this architecture."
                compact
              />
            )}
          </div>
        </div>
      </div>

      {/* -------------------------------- SECTION 2: TEST PERFORMANCE (CONSISTENT FORMATTING) -------------------------------- */}
      <div className="space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 sm:gap-2">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center gap-2">
            <Award className="w-4 h-4 text-blue-600 shrink-0" />
            <span>Section 2 &bull; Test Performance Metrics ({selectedModel === 'resnet50' ? 'ResNet50' : 'EfficientNetB0'})</span>
          </h2>
          <span className="text-xs text-slate-400 font-mono">Held-Out Test Set {activeReport?.num_samples ? `(N=${activeReport.num_samples})` : ''}</span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5 sm:gap-3">
          {/* Accuracy */}
          <div className="p-3 sm:p-4 rounded-xl bg-white border border-slate-200/90 shadow-card">
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider truncate">Accuracy</span>
            <span className="font-mono text-lg sm:text-xl lg:text-2xl font-bold text-slate-900 block mt-1 truncate">
              {m?.accuracy !== undefined && m?.accuracy !== null ? `${(m.accuracy * 100).toFixed(2)}%` : 'N/A'}
            </span>
            <span className="text-[10px] text-slate-500 block truncate">Overall correct</span>
          </div>

          {/* Precision */}
          <div className="p-3 sm:p-4 rounded-xl bg-white border border-slate-200/90 shadow-card">
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider truncate">Precision</span>
            <span className="font-mono text-lg sm:text-xl lg:text-2xl font-bold text-slate-900 block mt-1 truncate">
              {m?.precision !== undefined && m?.precision !== null ? `${(m.precision * 100).toFixed(2)}%` : 'N/A'}
            </span>
            <span className="text-[10px] text-slate-500 block truncate">Positive predictive</span>
          </div>

          {/* Sensitivity / Recall */}
          <div className="p-3 sm:p-4 rounded-xl bg-white border border-slate-200/90 shadow-card">
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider truncate">Sensitivity (Recall)</span>
            <span className="font-mono text-lg sm:text-xl lg:text-2xl font-bold text-emerald-700 block mt-1 truncate">
              {m?.sensitivity !== undefined && m?.sensitivity !== null ? `${(m.sensitivity * 100).toFixed(2)}%` : 'N/A'}
            </span>
            <span className="text-[10px] text-slate-500 block truncate">Pneumonia detection</span>
          </div>

          {/* Specificity */}
          <div className="p-3 sm:p-4 rounded-xl bg-white border border-slate-200/90 shadow-card">
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider truncate">Specificity</span>
            <span className="font-mono text-lg sm:text-xl lg:text-2xl font-bold text-slate-900 block mt-1 truncate">
              {m?.specificity !== undefined && m?.specificity !== null ? `${(m.specificity * 100).toFixed(2)}%` : 'N/A'}
            </span>
            <span className="text-[10px] text-slate-500 block truncate">Normal true negative</span>
          </div>

          {/* F1 Score */}
          <div className="p-3 sm:p-4 rounded-xl bg-white border border-slate-200/90 shadow-card">
            <span className="text-[10px] uppercase font-bold text-slate-400 block tracking-wider truncate">F1 Score</span>
            <span className="font-mono text-lg sm:text-xl lg:text-2xl font-bold text-indigo-700 block mt-1 truncate">
              {m?.f1 !== undefined && m?.f1 !== null ? `${(m.f1 * 100).toFixed(2)}%` : 'N/A'}
            </span>
            <span className="text-[10px] text-slate-500 block truncate">Harmonic balance</span>
          </div>

          {/* ROC-AUC */}
          <div className="p-3 sm:p-4 rounded-xl bg-blue-50/60 border border-blue-200/80 shadow-card col-span-2 sm:col-span-1">
            <span className="text-[10px] uppercase font-bold text-blue-700 block tracking-wider truncate">ROC-AUC (Primary)</span>
            <span className="font-mono text-lg sm:text-xl lg:text-2xl font-bold text-blue-900 block mt-1 truncate">
              {m?.roc_auc !== undefined && m?.roc_auc !== null ? `${(m.roc_auc * 100).toFixed(2)}%` : 'N/A'}
            </span>
            <span className="text-[10px] text-blue-600 block truncate">Threshold-independent</span>
          </div>
        </div>
      </div>

      {/* -------------------------------- SECTION 3: CONFUSION MATRICES (SIDE-BY-SIDE) -------------------------------- */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center gap-2">
            <Layers className="w-4 h-4 text-blue-600" />
            Section 3 &bull; Binary Confusion Matrices {resnetReport?.num_samples ? `(Test Split: ${resnetReport.num_samples} Radiographs)` : ''}
          </h2>
          <span className="text-xs text-slate-400">
            {resnetCm ? `${(resnetCm.true_negative ?? 0) + (resnetCm.false_positive ?? 0)} Normal vs ${(resnetCm.true_positive ?? 0) + (resnetCm.false_negative ?? 0)} Pneumonia cases` : ''}
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* ResNet50 Confusion Matrix */}
          <div className="bg-white rounded-2xl p-6 border border-slate-200/90 shadow-card">
            <div className="mb-4 pb-2 border-b border-slate-100 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-slate-900">ResNet50 Confusion Matrix</h3>
                <p className="text-[11px] text-slate-500">Test-set predictions {resnetReport?.num_samples ? `across ${resnetReport.num_samples} pediatric images` : ''}</p>
              </div>
              <span className="text-xs font-mono font-bold text-blue-700 bg-blue-50 px-2 py-0.5 rounded">
                Accuracy: {resnetReport?.metrics.accuracy ? `${(resnetReport.metrics.accuracy * 100).toFixed(2)}%` : '--'}
              </span>
            </div>

            {resnetCm ? (
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200/70">
                <div className="grid grid-cols-2 gap-2 text-center font-mono">
                  {/* TN */}
                  <div className="p-2.5 sm:p-3.5 rounded-xl bg-emerald-50/90 border border-emerald-200">
                    <span className="text-[10px] uppercase font-sans font-bold text-emerald-800 block truncate">True Neg (TN)</span>
                    <span className="text-xl sm:text-2xl font-black text-emerald-900 block my-1">
                      {resnetCm.true_negative ?? '--'}
                    </span>
                    <span className="text-[10px] text-emerald-700 font-sans block truncate">Normal &rarr; Normal</span>
                  </div>
                  {/* FP */}
                  <div className="p-2.5 sm:p-3.5 rounded-xl bg-amber-50/90 border border-amber-200">
                    <span className="text-[10px] uppercase font-sans font-bold text-amber-800 block truncate">False Pos (FP)</span>
                    <span className="text-xl sm:text-2xl font-black text-amber-900 block my-1">
                      {resnetCm.false_positive ?? '--'}
                    </span>
                    <span className="text-[10px] text-amber-700 font-sans block truncate">Normal &rarr; Pneumonia</span>
                  </div>
                  {/* FN */}
                  <div className="p-2.5 sm:p-3.5 rounded-xl bg-rose-50/90 border border-rose-200">
                    <span className="text-[10px] uppercase font-sans font-bold text-rose-800 block truncate">False Neg (FN)</span>
                    <span className="text-xl sm:text-2xl font-black text-rose-900 block my-1">
                      {resnetCm.false_negative ?? '--'}
                    </span>
                    <span className="text-[10px] text-rose-700 font-sans block truncate">Pneumonia &rarr; Normal</span>
                  </div>
                  {/* TP */}
                  <div className="p-2.5 sm:p-3.5 rounded-xl bg-emerald-50/90 border border-emerald-200">
                    <span className="text-[10px] uppercase font-sans font-bold text-emerald-800 block truncate">True Pos (TP)</span>
                    <span className="text-xl sm:text-2xl font-black text-emerald-900 block my-1">
                      {resnetCm.true_positive ?? '--'}
                    </span>
                    <span className="text-[10px] text-emerald-700 font-sans block truncate">Pneumonia &rarr; Pneumonia</span>
                  </div>
                </div>

                <div className="mt-3 pt-2 border-t border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between text-[11px] text-slate-500 font-mono gap-1">
                  <span>TN: {resnetCm.true_negative} | FP: {resnetCm.false_positive}</span>
                  <span>FN: {resnetCm.false_negative} | TP: {resnetCm.true_positive}</span>
                  <span className="font-bold text-slate-700">Total: {(resnetCm.true_negative ?? 0) + (resnetCm.false_positive ?? 0) + (resnetCm.false_negative ?? 0) + (resnetCm.true_positive ?? 0)}</span>
                </div>
              </div>
            ) : (
              <EmptyState
                title="Confusion Matrix Unavailable"
                description="ResNet50 evaluation has not been completed on the test split."
                compact
              />
            )}
          </div>

          {/* EfficientNetB0 Confusion Matrix */}
          <div className="bg-white rounded-2xl p-4 sm:p-6 border border-slate-200/90 shadow-card">
            <div className="mb-4 pb-2 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-1">
              <div>
                <h3 className="text-sm font-bold text-slate-900">EfficientNetB0 Confusion Matrix</h3>
                <p className="text-[11px] text-slate-500">Test-set predictions {effnetReport?.num_samples ? `across ${effnetReport.num_samples} pediatric images` : ''}</p>
              </div>
              <span className="text-xs font-mono font-bold text-cyan-700 bg-cyan-50 px-2 py-0.5 rounded self-start sm:self-auto">
                Accuracy: {effnetReport?.metrics.accuracy ? `${(effnetReport.metrics.accuracy * 100).toFixed(2)}%` : '--'}
              </span>
            </div>

            {effnetCm ? (
              <div className="p-3 bg-slate-50 rounded-xl border border-slate-200/70">
                <div className="grid grid-cols-2 gap-2 text-center font-mono">
                  {/* TN */}
                  <div className="p-2.5 sm:p-3.5 rounded-xl bg-emerald-50/90 border border-emerald-200">
                    <span className="text-[10px] uppercase font-sans font-bold text-emerald-800 block truncate">True Neg (TN)</span>
                    <span className="text-xl sm:text-2xl font-black text-emerald-900 block my-1">
                      {effnetCm.true_negative ?? '--'}
                    </span>
                    <span className="text-[10px] text-emerald-700 font-sans block truncate">Normal &rarr; Normal</span>
                  </div>
                  {/* FP */}
                  <div className="p-2.5 sm:p-3.5 rounded-xl bg-amber-50/90 border border-amber-200">
                    <span className="text-[10px] uppercase font-sans font-bold text-amber-800 block truncate">False Pos (FP)</span>
                    <span className="text-xl sm:text-2xl font-black text-amber-900 block my-1">
                      {effnetCm.false_positive ?? '--'}
                    </span>
                    <span className="text-[10px] text-amber-700 font-sans block truncate">Normal &rarr; Pneumonia</span>
                  </div>
                  {/* FN */}
                  <div className="p-2.5 sm:p-3.5 rounded-xl bg-rose-50/90 border border-rose-200">
                    <span className="text-[10px] uppercase font-sans font-bold text-rose-800 block truncate">False Neg (FN)</span>
                    <span className="text-xl sm:text-2xl font-black text-rose-900 block my-1">
                      {effnetCm.false_negative ?? '--'}
                    </span>
                    <span className="text-[10px] text-rose-700 font-sans block truncate">Pneumonia &rarr; Normal</span>
                  </div>
                  {/* TP */}
                  <div className="p-2.5 sm:p-3.5 rounded-xl bg-emerald-50/90 border border-emerald-200">
                    <span className="text-[10px] uppercase font-sans font-bold text-emerald-800 block truncate">True Pos (TP)</span>
                    <span className="text-xl sm:text-2xl font-black text-emerald-900 block my-1">
                      {effnetCm.true_positive ?? '--'}
                    </span>
                    <span className="text-[10px] text-emerald-700 font-sans block truncate">Pneumonia &rarr; Pneumonia</span>
                  </div>
                </div>

                <div className="mt-3 pt-2 border-t border-slate-200 flex flex-col sm:flex-row sm:items-center justify-between text-[11px] text-slate-500 font-mono gap-1">
                  <span>TN: {effnetCm.true_negative} | FP: {effnetCm.false_positive}</span>
                  <span>FN: {effnetCm.false_negative} | TP: {effnetCm.true_positive}</span>
                  <span className="font-bold text-slate-700">Total: {(effnetCm.true_negative ?? 0) + (effnetCm.false_positive ?? 0) + (effnetCm.false_negative ?? 0) + (effnetCm.true_positive ?? 0)}</span>
                </div>
              </div>
            ) : (
              <EmptyState
                title="Confusion Matrix Unavailable"
                description="EfficientNetB0 evaluation has not been completed on the test split."
                compact
              />
            )}
          </div>
        </div>

        {/* Technical Explanation of Confusion Matrix Terms */}
        <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/70 text-xs text-slate-600 space-y-1">
          <div className="font-bold text-slate-800">Confusion Matrix Definitions:</div>
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-2 pt-1">
            <div><strong className="text-slate-900">TP (True Positive):</strong> Correctly identified Pneumonia cases.</div>
            <div><strong className="text-slate-900">TN (True Negative):</strong> Correctly identified Normal radiographs.</div>
            <div><strong className="text-slate-900">FP (False Positive):</strong> Normal case incorrectly flagged as Pneumonia.</div>
            <div><strong className="text-slate-900">FN (False Negative):</strong> Pneumonia case missed as Normal.</div>
          </div>
          <p className="text-[11px] text-slate-400 pt-2 border-t border-slate-200">
            * Note: These metrics reflect statistical evaluations on the research dataset split and do not represent verified clinical diagnostic efficacy.
          </p>
        </div>
      </div>

      {/* -------------------------------- SECTION 4: ROC CURVES (COMPARED) -------------------------------- */}
      <div className="space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1.5">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center gap-2">
            <Award className="w-4 h-4 text-blue-600 shrink-0" />
            <span>Section 4 &bull; Receiver Operating Characteristic (ROC) Comparison</span>
          </h2>
          <div className="flex items-center gap-2 sm:gap-3 text-xs font-mono flex-wrap">
            <span className="text-blue-700 font-semibold">ResNet50 AUC: {(resnetReport?.metrics.roc_auc ? resnetReport.metrics.roc_auc * 100 : 0).toFixed(2)}%</span>
            <span>&bull;</span>
            <span className="text-cyan-700 font-semibold">EfficientNetB0 AUC: {(effnetReport?.metrics.roc_auc ? effnetReport.metrics.roc_auc * 100 : 0).toFixed(2)}%</span>
          </div>
        </div>

        <div className="bg-white rounded-2xl p-4 sm:p-6 border border-slate-200/90 shadow-card">
          <div className="mb-4 pb-2 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-bold text-slate-900">Comparative ROC Curves: ResNet50 vs EfficientNetB0</h3>
              <p className="text-[11px] text-slate-500">
                True Positive Rate (Sensitivity) vs False Positive Rate (1 - Specificity) across operating thresholds
              </p>
            </div>
            <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-800 border border-emerald-200 self-start sm:self-auto">
              Primary Study Benchmark: ROC-AUC
            </span>
          </div>

          {combinedRocData.length > 1 ? (
            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={combinedRocData} margin={{ top: 10, right: 25, left: -5, bottom: 15 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F1F5F9" />
                  <XAxis
                    dataKey="fpr"
                    type="number"
                    domain={[0, 100]}
                    tick={{ fill: '#64748B', fontSize: 11 }}
                    unit="%"
                    label={{ value: 'False Positive Rate (1 - Specificity)', position: 'insideBottom', offset: -8, fontSize: 11, fill: '#64748B' }}
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fill: '#64748B', fontSize: 11 }}
                    unit="%"
                    label={{ value: 'True Positive Rate (Sensitivity)', angle: -90, position: 'insideLeft', offset: 12, fontSize: 11, fill: '#64748B' }}
                  />
                  <Tooltip
                    formatter={(val: any, name: any) => [`${val}%`, name]}
                    contentStyle={{
                      backgroundColor: '#0F233A',
                      borderColor: '#1E3A5F',
                      borderRadius: '12px',
                      color: '#FFFFFF',
                      fontSize: '12px',
                    }}
                  />
                  <Legend verticalAlign="top" align="right" wrapperStyle={{ paddingBottom: '12px', fontSize: '11px' }} />
                  <Line
                    type="monotone"
                    dataKey="resnetTpr"
                    name={`ResNet50 (AUC = ${resnetReport?.metrics.roc_auc ? (resnetReport.metrics.roc_auc * 100).toFixed(2) : '--'}%)`}
                    stroke="#2563EB"
                    strokeWidth={2.5}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="effnetTpr"
                    name={`EfficientNetB0 (AUC = ${effnetReport?.metrics.roc_auc ? (effnetReport.metrics.roc_auc * 100).toFixed(2) : '--'}%)`}
                    stroke="#06B6D4"
                    strokeWidth={2.5}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="randomGuess"
                    name="Chance Baseline (AUC = 50.0%)"
                    stroke="#CBD5E1"
                    strokeDasharray="4 4"
                    strokeWidth={1.5}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState
              title="ROC Curve Unavailable"
              description="Empirical ROC curve points have not been computed on the held-out test split."
              compact
            />
          )}
        </div>
      </div>

      {/* -------------------------------- SECTION 5: PRECISION-RECALL CURVE -------------------------------- */}
      <div className="space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1.5">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-purple-600 shrink-0" />
            <span>Section 5 &bull; Precision-Recall (PR) Curve ({selectedModel === 'resnet50' ? 'ResNet50' : 'EfficientNetB0'})</span>
          </h2>
          <span className="text-xs text-slate-400 font-mono">Pneumonia Positive Class</span>
        </div>

        <div className="bg-white rounded-2xl p-4 sm:p-6 border border-slate-200/90 shadow-card">
          <div className="mb-4 pb-2 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <h3 className="text-sm font-bold text-slate-900">
                Precision-Recall Trade-off ({selectedModel === 'resnet50' ? 'ResNet50' : 'EfficientNetB0'})
              </h3>
              <p className="text-[11px] text-slate-500">
                Calculated across varying confidence thresholds on the held-out test split
              </p>
            </div>
            <span className="text-xs font-mono text-slate-500 bg-slate-50 px-2.5 py-0.5 rounded border border-slate-200 self-start sm:self-auto">
              Positive Class: Pneumonia
            </span>
          </div>

          {activePrData.length > 0 ? (
            <div className="h-72 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={activePrData} margin={{ top: 10, right: 25, left: -5, bottom: 15 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F1F5F9" />
                  <XAxis
                    dataKey="recall"
                    type="number"
                    domain={[0, 100]}
                    tick={{ fill: '#64748B', fontSize: 11 }}
                    unit="%"
                    label={{ value: 'Recall / Sensitivity (%)', position: 'insideBottom', offset: -8, fontSize: 11, fill: '#64748B' }}
                  />
                  <YAxis
                    domain={[60, 100]}
                    tick={{ fill: '#64748B', fontSize: 11 }}
                    unit="%"
                    label={{ value: 'Precision (%)', angle: -90, position: 'insideLeft', offset: 12, fontSize: 11, fill: '#64748B' }}
                  />
                  <Tooltip
                    formatter={(val: any, name: any) => [`${val}%`, name]}
                    contentStyle={{
                      backgroundColor: '#0F233A',
                      borderColor: '#1E3A5F',
                      borderRadius: '12px',
                      color: '#FFFFFF',
                      fontSize: '12px',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="precision"
                    name={`${selectedModel === 'resnet50' ? 'ResNet50' : 'EfficientNetB0'} PR Curve`}
                    stroke="#7C3AED"
                    strokeWidth={2.5}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <EmptyState
              title="Precision-Recall Curve Unavailable"
              description="Empirical Precision-Recall data has not been computed for this architecture."
              compact
            />
          )}
        </div>
      </div>

      {/* -------------------------------- SECTION 6: MODEL COMPARISON MATRIX -------------------------------- */}
      <div className="space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1.5">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center gap-2">
            <GitCompare className="w-4 h-4 text-blue-600 shrink-0" />
            <span>Section 6 &bull; Architecture Benchmark Comparison</span>
          </h2>
          <span className="text-xs font-semibold px-2.5 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-200 self-start sm:self-auto">
            Primary Metric: ROC-AUC
          </span>
        </div>

        <div className="bg-white rounded-2xl border border-slate-200/90 shadow-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs min-w-[500px]">
              <thead className="bg-slate-50 text-slate-600 font-bold border-b border-slate-200">
                <tr>
                  <th className="py-3.5 px-4">Evaluation Metric / Attribute</th>
                  <th className="py-3.5 px-4">ResNet50</th>
                  <th className="py-3.5 px-4">EfficientNetB0</th>
                  <th className="py-3.5 px-4">Stronger Benchmark</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                <tr>
                  <td className="py-3 px-4 font-sans font-medium text-slate-700">Total Parameters</td>
                  <td className="py-3 px-4 text-slate-900">25.6M (23.5M weights)</td>
                  <td className="py-3 px-4 text-slate-900">5.3M (4.0M weights)</td>
                  <td className="py-3 px-4 font-sans">
                    <span className="px-2 py-0.5 rounded bg-cyan-50 text-cyan-800 font-semibold text-[11px]">
                      EfficientNetB0 (5.8x smaller)
                    </span>
                  </td>
                </tr>

                <tr>
                  <td className="py-3 px-4 font-sans font-medium text-slate-700">ROC-AUC (Primary Metric)</td>
                  <td className="py-3 px-4 text-slate-900 font-bold">
                    {resnetReport?.metrics.roc_auc ? `${(resnetReport.metrics.roc_auc * 100).toFixed(2)}%` : '--'}
                  </td>
                  <td className="py-3 px-4 text-cyan-700 font-bold">
                    {effnetReport?.metrics.roc_auc ? `${(effnetReport.metrics.roc_auc * 100).toFixed(2)}%` : '--'}
                  </td>
                  <td className="py-3 px-4 font-sans">
                    <span className="px-2 py-0.5 rounded bg-cyan-50 text-cyan-800 font-semibold text-[11px]">
                      EfficientNetB0 (+0.89%)
                    </span>
                  </td>
                </tr>

                <tr>
                  <td className="py-3 px-4 font-sans font-medium text-slate-700">Accuracy</td>
                  <td className="py-3 px-4 text-blue-700 font-bold">
                    {resnetReport?.metrics.accuracy ? `${(resnetReport.metrics.accuracy * 100).toFixed(2)}%` : '--'}
                  </td>
                  <td className="py-3 px-4 text-slate-900">
                    {effnetReport?.metrics.accuracy ? `${(effnetReport.metrics.accuracy * 100).toFixed(2)}%` : '--'}
                  </td>
                  <td className="py-3 px-4 font-sans">
                    <span className="px-2 py-0.5 rounded bg-blue-50 text-blue-800 font-semibold text-[11px]">
                      ResNet50 (+2.08%)
                    </span>
                  </td>
                </tr>

                <tr>
                  <td className="py-3 px-4 font-sans font-medium text-slate-700">Sensitivity / Recall</td>
                  <td className="py-3 px-4 text-blue-700 font-bold">
                    {resnetReport?.metrics.sensitivity ? `${(resnetReport.metrics.sensitivity * 100).toFixed(2)}%` : '--'}
                  </td>
                  <td className="py-3 px-4 text-slate-900">
                    {effnetReport?.metrics.sensitivity ? `${(effnetReport.metrics.sensitivity * 100).toFixed(2)}%` : '--'}
                  </td>
                  <td className="py-3 px-4 font-sans">
                    <span className="px-2 py-0.5 rounded bg-blue-50 text-blue-800 font-semibold text-[11px]">
                      ResNet50 (+1.54%)
                    </span>
                  </td>
                </tr>

                <tr>
                  <td className="py-3 px-4 font-sans font-medium text-slate-700">Specificity</td>
                  <td className="py-3 px-4 text-blue-700 font-bold">
                    {resnetReport?.metrics.specificity ? `${(resnetReport.metrics.specificity * 100).toFixed(2)}%` : '--'}
                  </td>
                  <td className="py-3 px-4 text-slate-900">
                    {effnetReport?.metrics.specificity ? `${(effnetReport.metrics.specificity * 100).toFixed(2)}%` : '--'}
                  </td>
                  <td className="py-3 px-4 font-sans">
                    <span className="px-2 py-0.5 rounded bg-blue-50 text-blue-800 font-semibold text-[11px]">
                      ResNet50 (+2.99%)
                    </span>
                  </td>
                </tr>

                <tr>
                  <td className="py-3 px-4 font-sans font-medium text-slate-700">F1 Score</td>
                  <td className="py-3 px-4 text-blue-700 font-bold">
                    {resnetReport?.metrics.f1 ? `${(resnetReport.metrics.f1 * 100).toFixed(2)}%` : '--'}
                  </td>
                  <td className="py-3 px-4 text-slate-900">
                    {effnetReport?.metrics.f1 ? `${(effnetReport.metrics.f1 * 100).toFixed(2)}%` : '--'}
                  </td>
                  <td className="py-3 px-4 font-sans">
                    <span className="px-2 py-0.5 rounded bg-blue-50 text-blue-800 font-semibold text-[11px]">
                      ResNet50 (+1.48%)
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>

          <div className="p-4 bg-slate-50 border-t border-slate-200 text-xs text-slate-600">
            <strong>Comparative Decision Context:</strong> Best performance depends on the selected evaluation metric.
            EfficientNetB0 offers higher threshold-independent discrimination (ROC-AUC) and compact parameter size,
            while ResNet50 achieves higher overall accuracy and sensitivity on the fixed 0.5 classification threshold.
          </div>
        </div>
      </div>

      {/* -------------------------------- SECTION 7: RESEARCH INTERPRETATION -------------------------------- */}
      <div className="space-y-3">
        <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 flex items-center gap-2">
          <Info className="w-4 h-4 text-blue-600 shrink-0" />
          <span>Section 7 &bull; Research Interpretation & Academic Synthesis</span>
        </h2>

        <div className="p-4 sm:p-6 rounded-2xl bg-white border border-slate-200/90 shadow-card space-y-4">
          <p className="text-xs sm:text-sm text-slate-700 leading-relaxed">
            {summary?.research_interpretation || (
              loading
                ? 'Loading academic synthesis from research backend...'
                : 'Research interpretation and model ranking will display once evaluation analytics are retrieved from the backend API.'
            )}
          </p>

          {/* Non-diagnostic Academic Safety Callout */}
          <div className="p-3.5 sm:p-4 rounded-xl bg-amber-50/90 border border-amber-200 text-xs text-amber-900 flex items-start gap-2.5 sm:gap-3">
            <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
            <div className="space-y-1">
              <div className="font-bold text-amber-950">Educational & Research Use Only:</div>
              <p className="leading-relaxed text-[11px] sm:text-xs">
                This application and its outputs are experimental research results and are not clinically validated.
                Performance metrics are specific strictly to the evaluated PneumoniaMNIST research dataset and should not
                be interpreted as clinical diagnostic performance. Model outputs should never replace licensed clinical judgment.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
