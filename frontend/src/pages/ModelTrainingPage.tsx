import React, { useState, useEffect, useRef } from 'react';
import {
  Cpu,
  Play,
  Square,
  CheckCircle2,
  AlertCircle,
  Sliders,
  Clock,
  Award,
  TrendingDown,
  TrendingUp,
  RefreshCw,
  ArrowRight,
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import type { ModelId, TrainingJob } from '../types';
import { medicalApi } from '../services/api';
import { StatusBadge } from '../components/common/StatusBadge';

export const ModelTrainingPage: React.FC = () => {
  const navigate = useNavigate();

  // Configuration state
  const [selectedModel, setSelectedModel] = useState<ModelId>('efficientnet_b0');
  const [epochs, setEpochs] = useState<number>(5);
  const [batchSize, setBatchSize] = useState<number>(32);
  const [learningRate, setLearningRate] = useState<string>('0.001');
  const [optimizer, setOptimizer] = useState<'Adam' | 'AdamW'>('AdamW');
  const [fineTuning, setFineTuning] = useState<boolean>(true);

  // Training execution state
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [feedbackMessage, setFeedbackMessage] = useState<{
    type: 'success' | 'error' | 'info';
    text: string;
  } | null>(null);

  // Telemetry state
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [loadingJobs, setLoadingJobs] = useState<boolean>(true);

  const pollIntervalRef = useRef<number | null>(null);

  // Fetch all training jobs from real backend
  const fetchTrainingStatus = async () => {
    try {
      const fetchedJobs = await medicalApi.getTrainingStatus();
      setJobs(fetchedJobs);
    } catch (err: unknown) {
      console.error('Failed to fetch training jobs:', err);
    } finally {
      setLoadingJobs(false);
    }
  };

  useEffect(() => {
    fetchTrainingStatus();
  }, []);

  // Poll while any job is actively running
  const isAnyRunning = jobs.some((j) => j.status === 'training' || (j.status as any) === 'running');
  const activeJob = jobs.find((j) => j.modelId === selectedModel);

  useEffect(() => {
    if (isAnyRunning) {
      pollIntervalRef.current = window.setInterval(() => {
        fetchTrainingStatus();
      }, 1500);
    } else if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, [isAnyRunning]);

  const handleStartTraining = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setFeedbackMessage(null);

    const lrNum = parseFloat(learningRate);
    if (isNaN(lrNum) || lrNum <= 0) {
      setFeedbackMessage({
        type: 'error',
        text: 'Invalid learning rate. Please enter a positive number (e.g., 0.001).',
      });
      setIsSubmitting(false);
      return;
    }

    try {
      await medicalApi.startTraining({
        model_name: selectedModel,
        epochs: Number(epochs),
        batch_size: Number(batchSize),
        learning_rate: lrNum,
        optimizer,
        fine_tune: fineTuning,
      });

      setFeedbackMessage({
        type: 'success',
        text: `Training initiated for ${
          selectedModel === 'resnet50' ? 'ResNet50' : 'EfficientNetB0'
        } on PneumoniaMNIST. Monitoring live epoch telemetry...`,
      });

      await fetchTrainingStatus();
    } catch (err: any) {
      const detail = err.response?.data?.detail || err.message || 'Failed to start training.';
      setFeedbackMessage({
        type: 'error',
        text: detail,
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleStopTraining = async () => {
    if (!activeJob) return;
    try {
      await medicalApi.stopTraining(selectedModel);
      setFeedbackMessage({
        type: 'info',
        text: `Stop signal sent for ${selectedModel}. Training will halt at the next batch/epoch boundary.`,
      });
      await fetchTrainingStatus();
    } catch (err: any) {
      setFeedbackMessage({
        type: 'error',
        text: `Failed to stop training: ${err.message}`,
      });
    }
  };

  const isModelTraining = activeJob?.status === 'training' || (activeJob?.status as any) === 'running';

  return (
    <div className="space-y-6 animate-in fade-in duration-200">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div>
          <h2 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight flex items-center gap-2.5">
            <Cpu className="w-6 h-6 text-blue-600" />
            Model Training & Fine-Tuning
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Real PyTorch transfer learning engine executing on PneumoniaMNIST (CPU Mode)
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => fetchTrainingStatus()}
            className="p-2 rounded-xl border border-slate-200 text-slate-600 hover:text-slate-900 hover:bg-slate-50 text-xs font-medium flex items-center gap-1.5 transition-colors"
            title="Refresh Status"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loadingJobs ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
          <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold bg-blue-50 text-blue-800 border border-blue-200">
            <span className="w-2 h-2 rounded-full bg-blue-600 animate-pulse" />
            <span>Active PyTorch Engine</span>
          </span>
        </div>
      </div>

      {/* Feedback Alert */}
      {feedbackMessage && (
        <div
          className={`p-4 rounded-xl border text-xs flex items-start gap-2.5 animate-in fade-in duration-200 ${
            feedbackMessage.type === 'success'
              ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
              : feedbackMessage.type === 'error'
              ? 'bg-red-50 border-red-200 text-red-800'
              : 'bg-blue-50 border-blue-200 text-blue-800'
          }`}
        >
          {feedbackMessage.type === 'success' ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
          ) : (
            <AlertCircle className="w-4 h-4 text-red-600 shrink-0 mt-0.5" />
          )}
          <div className="flex-1 font-medium">{feedbackMessage.text}</div>
        </div>
      )}

      {/* Main Grid: Form & Live Telemetry */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Configuration Form (2 Cols) */}
        <div className="lg:col-span-2 bg-white rounded-2xl p-4 sm:p-6 border border-slate-200/90 shadow-card">
          <div className="mb-5 sm:mb-6 pb-4 border-b border-slate-100 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <h3 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-blue-600" />
              Training Configuration
            </h3>
            <span className="text-xs font-mono text-slate-500 bg-slate-100 px-2 py-0.5 rounded-md self-start sm:self-auto">
              PneumoniaMNIST (Train: 4,708 &bull; Val: 524)
            </span>
          </div>

          <form onSubmit={handleStartTraining} className="space-y-6">
            {/* Architecture Selection */}
            <div>
              <label className="block text-xs font-bold uppercase tracking-wider text-slate-600 mb-3">
                Select Architecture
              </label>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <label
                  className={`p-4 rounded-xl border cursor-pointer transition-all ${
                    selectedModel === 'resnet50'
                      ? 'border-blue-600 bg-blue-50/50 ring-2 ring-blue-600/20'
                      : 'border-slate-200 hover:border-slate-300 bg-white'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <input
                      type="radio"
                      name="model"
                      value="resnet50"
                      checked={selectedModel === 'resnet50'}
                      onChange={() => setSelectedModel('resnet50')}
                      disabled={isAnyRunning}
                      className="text-blue-600 focus:ring-blue-500"
                    />
                    <div>
                      <span className="text-sm font-bold text-slate-900 block">ResNet50</span>
                      <span className="text-xs text-slate-500">23.5M Params &bull; Bottleneck ResNet</span>
                    </div>
                  </div>
                </label>

                <label
                  className={`p-4 rounded-xl border cursor-pointer transition-all ${
                    selectedModel === 'efficientnet_b0'
                      ? 'border-cyan-600 bg-cyan-50/50 ring-2 ring-cyan-600/20'
                      : 'border-slate-200 hover:border-slate-300 bg-white'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <input
                      type="radio"
                      name="model"
                      value="efficientnet_b0"
                      checked={selectedModel === 'efficientnet_b0'}
                      onChange={() => setSelectedModel('efficientnet_b0')}
                      disabled={isAnyRunning}
                      className="text-cyan-600 focus:ring-cyan-500"
                    />
                    <div>
                      <span className="text-sm font-bold text-slate-900 block">EfficientNetB0</span>
                      <span className="text-xs text-slate-500">4.0M Params &bull; MBConv Scaling</span>
                    </div>
                  </div>
                </label>
              </div>
            </div>

            {/* Hyperparameters Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Epochs
                </label>
                <input
                  type="number"
                  min="1"
                  max="50"
                  value={epochs}
                  onChange={(e) => setEpochs(Number(e.target.value))}
                  disabled={isAnyRunning}
                  className="w-full px-3.5 py-2.5 rounded-xl border border-slate-200 text-sm bg-slate-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 font-mono disabled:opacity-60 min-h-[44px]"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Batch Size
                </label>
                <select
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  disabled={isAnyRunning}
                  className="w-full px-3.5 py-2.5 rounded-xl border border-slate-200 text-sm bg-slate-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 font-mono disabled:opacity-60 min-h-[44px]"
                >
                  <option value={16}>16</option>
                  <option value={32}>32 (Optimal for CPU)</option>
                  <option value={64}>64</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Learning Rate
                </label>
                <input
                  type="text"
                  value={learningRate}
                  onChange={(e) => setLearningRate(e.target.value)}
                  disabled={isAnyRunning}
                  className="w-full px-3.5 py-2.5 rounded-xl border border-slate-200 text-sm bg-slate-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 font-mono disabled:opacity-60 min-h-[44px]"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-700 mb-1.5">
                  Optimizer
                </label>
                <select
                  value={optimizer}
                  onChange={(e) => setOptimizer(e.target.value as 'Adam' | 'AdamW')}
                  disabled={isAnyRunning}
                  className="w-full px-3.5 py-2.5 rounded-xl border border-slate-200 text-sm bg-slate-50 focus:bg-white focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 font-mono disabled:opacity-60 min-h-[44px]"
                >
                  <option value="AdamW">AdamW (Decoupled Weight Decay)</option>
                  <option value="Adam">Adam</option>
                </select>
              </div>
            </div>

            {/* Fine Tuning Toggle */}
            <div className="p-4 rounded-xl border border-slate-200 bg-slate-50/50 flex items-center justify-between">
              <div>
                <span className="text-xs font-bold text-slate-900 block">Fine-Tuning Mode</span>
                <span className="text-[11px] text-slate-500">
                  Unfreezes backbone with conservative differential learning rate (0.1x) for radiograph adaptation
                </span>
              </div>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={fineTuning}
                  onChange={(e) => setFineTuning(e.target.checked)}
                  disabled={isAnyRunning}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-slate-300 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-slate-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600 peer-disabled:opacity-60" />
              </label>
            </div>

            {/* Submit / Action Buttons */}
            <div className="flex items-center gap-3">
              {isModelTraining ? (
                <button
                  type="button"
                  onClick={handleStopTraining}
                  className="w-full py-3 px-4 rounded-xl text-sm font-semibold text-white bg-red-600 hover:bg-red-700 active:bg-red-800 shadow-sm flex items-center justify-center gap-2 transition-all"
                >
                  <Square className="w-4 h-4 fill-white" />
                  <span>Stop Active Training</span>
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={isSubmitting || isAnyRunning}
                  className="w-full py-3.5 px-4 rounded-xl text-sm font-semibold text-white bg-slate-900 hover:bg-blue-700 active:bg-blue-800 shadow-md hover:shadow-glow-blue transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isSubmitting ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : (
                    <Play className="w-4 h-4 fill-white" />
                  )}
                  <span>
                    {isAnyRunning
                      ? 'Another Job Running...'
                      : `Launch Real Training for ${selectedModel === 'resnet50' ? 'ResNet50' : 'EfficientNetB0'}`}
                  </span>
                </button>
              )}
            </div>
          </form>
        </div>

        {/* Right: Live Telemetry & Job Status Card */}
        <div className="space-y-4">
          <div className="p-5 rounded-2xl bg-white border border-slate-200/90 shadow-card">
            <div className="flex items-center justify-between pb-3 border-b border-slate-100 mb-4">
              <div>
                <h4 className="text-sm font-bold text-slate-900 flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-blue-600" />
                  {activeJob?.modelName || (selectedModel === 'resnet50' ? 'ResNet50' : 'EfficientNetB0')}
                </h4>
                <span className="text-[11px] text-slate-500">Live Status & Checkpoint</span>
              </div>
              {activeJob && <StatusBadge status={activeJob.status} size="sm" />}
            </div>

            {activeJob ? (
              <div className="space-y-4">
                {/* Progress Bar */}
                <div>
                  <div className="flex justify-between text-xs text-slate-600 mb-1 font-medium">
                    <span>
                      Epoch {activeJob.currentEpoch} / {activeJob.totalEpochs}
                    </span>
                    <span className="font-mono font-bold text-blue-700">{activeJob.progress}%</span>
                  </div>
                  <div className="w-full h-2.5 rounded-full bg-slate-100 overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 rounded-full ${
                        activeJob.status === 'completed'
                          ? 'bg-emerald-500'
                          : activeJob.status === 'failed'
                          ? 'bg-red-500'
                          : 'bg-blue-600'
                      }`}
                      style={{ width: `${activeJob.progress}%` }}
                    />
                  </div>
                </div>

                {/* Metric Summary Grid */}
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
                    <span className="text-[10px] text-slate-400 block font-medium flex items-center gap-1">
                      <Award className="w-3 h-3 text-amber-500" />
                      Best Val Acc
                    </span>
                    <span className="font-mono font-bold text-slate-900 text-sm">
                      {activeJob.bestValScore !== null
                        ? `${(activeJob.bestValScore * 100).toFixed(1)}%`
                        : '--'}
                    </span>
                    {activeJob.bestEpoch && (
                      <span className="text-[10px] text-slate-500 block">@ Epoch {activeJob.bestEpoch}</span>
                    )}
                  </div>

                  <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
                    <span className="text-[10px] text-slate-400 block font-medium flex items-center gap-1">
                      <Clock className="w-3 h-3 text-slate-400" />
                      Duration
                    </span>
                    <span className="font-mono font-bold text-slate-900 text-sm">
                      {activeJob.duration || '--'}
                    </span>
                    <span className="text-[10px] text-slate-500 block">PyTorch CPU</span>
                  </div>

                  {activeJob.currentTrainLoss !== undefined && activeJob.currentTrainLoss !== null && (
                    <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
                      <span className="text-[10px] text-slate-400 block font-medium flex items-center gap-1">
                        <TrendingDown className="w-3 h-3 text-indigo-500" />
                        Train Loss
                      </span>
                      <span className="font-mono font-bold text-slate-900 text-sm">
                        {activeJob.currentTrainLoss.toFixed(4)}
                      </span>
                    </div>
                  )}

                  {activeJob.currentValLoss !== undefined && activeJob.currentValLoss !== null && (
                    <div className="p-2.5 rounded-xl bg-slate-50 border border-slate-100">
                      <span className="text-[10px] text-slate-400 block font-medium flex items-center gap-1">
                        <TrendingUp className="w-3 h-3 text-emerald-500" />
                        Val Loss
                      </span>
                      <span className="font-mono font-bold text-slate-900 text-sm">
                        {activeJob.currentValLoss.toFixed(4)}
                      </span>
                    </div>
                  )}
                </div>

                {/* Checkpoint Badge */}
                {activeJob.checkpointSaved && (
                  <div className="p-3 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs">
                    <div className="flex items-center gap-1.5 font-bold mb-1">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                      <span>Best Checkpoint Saved</span>
                    </div>
                    <p className="font-mono text-[10px] text-emerald-700 truncate">
                      {activeJob.checkpointPath || `${selectedModel}_best.pt`}
                    </p>
                  </div>
                )}

                {activeJob.status === 'completed' && (
                  <button
                    onClick={() => navigate('/comparison')}
                    className="w-full py-2 px-3 rounded-xl text-xs font-semibold text-blue-700 bg-blue-50 hover:bg-blue-100 border border-blue-200 flex items-center justify-center gap-1.5 transition-colors"
                  >
                    <span>View in Model Comparison</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            ) : (
              <div className="text-center py-6 text-xs text-slate-400">
                No training job initialized for this architecture yet.
              </div>
            )}
          </div>

          {/* Educational Note */}
          <div className="p-4 rounded-2xl bg-amber-50/70 border border-amber-200/80 text-xs text-amber-900 space-y-1">
            <span className="font-bold">Medical Research Disclaimer: </span>
            <p className="text-[11px] leading-relaxed text-amber-800">
              Transfer learning training runs on PneumoniaMNIST pediatric radiographs are strictly for educational and academic research. Model checkpoints are not for clinical diagnostic usage.
            </p>
          </div>
        </div>
      </div>

      {/* Epoch Telemetry Table */}
      {activeJob?.history && activeJob.history.length > 0 && (
        <div className="bg-white rounded-2xl p-6 border border-slate-200/90 shadow-card">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-100">
            <div>
              <h3 className="text-base font-bold text-slate-900 tracking-tight">
                Epoch Training History & Telemetry
              </h3>
              <p className="text-xs text-slate-500">
                Live convergence tracking across train and validation splits
              </p>
            </div>
            <span className="text-xs font-mono text-slate-400">
              {activeJob.history.length} Epochs Recorded
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 text-slate-500 uppercase tracking-wider text-[10px] font-semibold">
                <tr>
                  <th className="py-2.5 px-3 rounded-l-lg">Epoch</th>
                  <th className="py-2.5 px-3">Train Loss</th>
                  <th className="py-2.5 px-3">Train Accuracy</th>
                  <th className="py-2.5 px-3">Val Loss</th>
                  <th className="py-2.5 px-3">Val Accuracy</th>
                  <th className="py-2.5 px-3 rounded-r-lg">Duration</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {activeJob.history.map((metric) => {
                  const isBest = metric.epoch === activeJob.bestEpoch;
                  return (
                    <tr
                      key={metric.epoch}
                      className={isBest ? 'bg-emerald-50/50 font-semibold' : 'hover:bg-slate-50/60'}
                    >
                      <td className="py-2.5 px-3 text-slate-800">
                        Epoch {metric.epoch}
                        {isBest && (
                          <span className="ml-2 inline-flex items-center text-[10px] text-emerald-700 bg-emerald-100 px-1.5 py-0.2 rounded-md font-sans">
                            Best
                          </span>
                        )}
                      </td>
                      <td className="py-2.5 px-3 text-slate-600">{metric.train_loss.toFixed(4)}</td>
                      <td className="py-2.5 px-3 text-slate-800">
                        {(metric.train_acc * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-3 text-slate-600">{metric.val_loss.toFixed(4)}</td>
                      <td className="py-2.5 px-3 text-slate-900 font-bold">
                        {(metric.val_acc * 100).toFixed(1)}%
                      </td>
                      <td className="py-2.5 px-3 text-slate-500">
                        {metric.epoch_duration_seconds}s
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
