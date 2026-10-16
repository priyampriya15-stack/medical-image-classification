import React, { useState, useEffect } from 'react';
import { Database, ShieldCheck, AlertCircle, RefreshCw } from 'lucide-react';
import { medicalApi } from '../services/api';
import type { DatasetInfo, DatasetSample } from '../types';
import { StatusBadge } from '../components/common/StatusBadge';
import { CardSkeleton } from '../components/common/LoadingSkeleton';
import { EmptyState } from '../components/common/EmptyState';

export const DatasetExplorerPage: React.FC = () => {
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [samples, setSamples] = useState<DatasetSample[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filterClass, setFilterClass] = useState<'All' | 'Normal' | 'Pneumonia'>('All');

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [info, sampleList] = await Promise.all([
        medicalApi.getDatasetInfo(),
        medicalApi.getDatasetSamples(),
      ]);
      setDatasetInfo(info);
      setSamples(sampleList);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to connect to backend dataset service');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const filtered = samples.filter((s) => {
    if (filterClass === 'All') return true;
    return s.classLabel === filterClass;
  });

  return (
    <div className="space-y-6 animate-in fade-in duration-200">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight flex items-center gap-2.5">
            <Database className="w-6 h-6 text-blue-600" />
            Dataset Explorer &bull; {datasetInfo ? datasetInfo.name : 'PneumoniaMNIST'}
          </h2>
          <p className="text-xs text-slate-500 mt-1">
            Standardized pediatric chest radiograph dataset curated for binary classification benchmarking
          </p>
        </div>

        <div className="flex items-center gap-3">
          {datasetInfo?.available && (
            <span className="text-xs font-mono bg-blue-50 text-blue-700 px-3 py-1.5 rounded-xl border border-blue-200/80 font-semibold">
              Total: {datasetInfo.total_samples.toLocaleString()} Images
            </span>
          )}
          <button
            onClick={loadData}
            disabled={loading}
            className="p-2 rounded-xl text-slate-600 hover:text-slate-900 hover:bg-slate-100 transition-colors border border-slate-200/80"
            title="Reload dataset info"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin text-blue-600' : ''}`} />
          </button>
        </div>
      </div>

      {/* Error state */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
            <span>{error}</span>
          </div>
          <button
            onClick={loadData}
            className="text-xs font-semibold text-rose-700 hover:underline ml-4 shrink-0"
          >
            Retry Connection
          </button>
        </div>
      )}

      {/* Empty State when dataset is unavailable */}
      {!loading && datasetInfo && !datasetInfo.available && (
        <EmptyState
          icon={Database}
          title="Dataset not available"
          description={datasetInfo.status_message || "The PneumoniaMNIST dataset archive ('pneumoniamnist.npz') is not found in the backend data directory."}
          actionText="Retry Loading"
          onAction={loadData}
        />
      )}

      {/* Dataset Summary Cards */}
      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <CardSkeleton key={i} className="h-28" />
          ))}
        </div>
      ) : datasetInfo && datasetInfo.available ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 rounded-2xl bg-white border border-slate-200/80 shadow-card">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Training Split</span>
            <h3 className="text-xl font-bold text-slate-900 mt-1">
              {datasetInfo.splits.train.toLocaleString()} samples
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              {((datasetInfo.splits.train / datasetInfo.total_samples) * 100).toFixed(1)}% of total dataset
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-white border border-slate-200/80 shadow-card">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Validation Split</span>
            <h3 className="text-xl font-bold text-slate-900 mt-1">
              {datasetInfo.splits.validation.toLocaleString()} samples
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              {((datasetInfo.splits.validation / datasetInfo.total_samples) * 100).toFixed(1)}% of total dataset
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-white border border-slate-200/80 shadow-card">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Test Split</span>
            <h3 className="text-xl font-bold text-slate-900 mt-1">
              {datasetInfo.splits.test.toLocaleString()} samples
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              {((datasetInfo.splits.test / datasetInfo.total_samples) * 100).toFixed(1)}% evaluation partition
            </p>
          </div>

          <div className="p-4 rounded-2xl bg-white border border-slate-200/80 shadow-card">
            <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Class Balance</span>
            <h3 className="text-sm font-bold text-slate-900 mt-1">
              Normal: {datasetInfo.class_distribution.Normal?.toLocaleString() ?? 0}
            </h3>
            <p className="text-xs text-slate-600 mt-0.5 font-medium">
              Pneumonia: {datasetInfo.class_distribution.Pneumonia?.toLocaleString() ?? 0}
            </p>
          </div>
        </div>
      ) : null}

      {/* Dataset Specifications & Dimensions Card */}
      {datasetInfo && datasetInfo.available && (
        <div className="p-4 sm:p-5 rounded-2xl bg-white border border-slate-200/80 shadow-card flex flex-col lg:flex-row lg:items-center justify-between gap-4 text-xs">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4 w-full lg:w-auto">
            <div className="p-2 sm:p-0 rounded-lg bg-slate-50 sm:bg-transparent">
              <span className="text-slate-400 font-semibold block uppercase text-[10px]">Image Dimensions</span>
              <span className="font-mono font-bold text-slate-800 text-xs sm:text-sm">{datasetInfo.image_dimensions}</span>
            </div>
            <div className="p-2 sm:p-0 rounded-lg bg-slate-50 sm:bg-transparent">
              <span className="text-slate-400 font-semibold block uppercase text-[10px]">Target Classes ({datasetInfo.num_classes})</span>
              <span className="font-semibold text-slate-800 text-xs sm:text-sm">{datasetInfo.classes.join(', ')}</span>
            </div>
            <div className="p-2 sm:p-0 rounded-lg bg-slate-50 sm:bg-transparent">
              <span className="text-slate-400 font-semibold block uppercase text-[10px]">Data Source</span>
              <span className="text-slate-700 text-xs sm:text-sm truncate block">MedMNIST v2 (Zenodo #6496656)</span>
            </div>
          </div>

          <div className="text-xs text-slate-500 flex items-center gap-1.5 pt-2 lg:pt-0 border-t lg:border-t-0 border-slate-100 shrink-0">
            <ShieldCheck className="w-4 h-4 text-emerald-600 shrink-0" />
            <span>Strict zero-leakage partitions &bull; Anonymized radiographs</span>
          </div>
        </div>
      )}

      {/* Filter and Radiograph Grid */}
      <div className="p-4 rounded-2xl bg-white border border-slate-200/80 shadow-card flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 sm:gap-4">
        <div className="flex flex-wrap items-center gap-2 w-full sm:w-auto">
          <span className="text-xs font-semibold text-slate-600 mr-1">Filter Class:</span>
          {(['All', 'Normal', 'Pneumonia'] as const).map((label) => (
            <button
              key={label}
              onClick={() => setFilterClass(label)}
              className={`flex-1 sm:flex-initial px-3.5 py-2 sm:py-1.5 rounded-xl text-xs font-semibold transition-all min-h-[38px] flex items-center justify-center ${
                filterClass === label
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="text-xs text-slate-400 self-end sm:self-auto font-mono">
          Showing {filtered.length} sample previews
        </div>
      </div>

      {/* Image Grid */}
      {loading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <CardSkeleton key={i} className="h-48" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {filtered.map((item) => (
            <div
              key={item.id}
              className="bg-white rounded-2xl overflow-hidden border border-slate-200/80 shadow-card hover:shadow-card-hover transition-all"
            >
              <div className="h-48 sm:h-44 bg-slate-950 relative overflow-hidden flex items-center justify-center">
                <img
                  src={item.imageUrl}
                  alt={item.id}
                  className="w-full h-full object-contain filter contrast-125"
                />
                <span className="absolute top-2.5 left-2.5 bg-slate-900/80 text-white text-[10px] font-mono px-2 py-0.5 rounded border border-slate-700 backdrop-blur-xs">
                  {item.split.toUpperCase()}
                </span>
              </div>
              <div className="p-3.5 flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <span className="text-[11px] font-mono text-slate-700 font-semibold block truncate">
                    {item.id}
                  </span>
                  <span className="text-[10px] text-slate-400 block font-mono">
                    {item.dimensions}
                  </span>
                </div>
                <StatusBadge status={item.classLabel} size="sm" showDot={false} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
