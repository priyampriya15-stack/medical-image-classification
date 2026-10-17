import React, { useState, useRef } from 'react';
import {
  Upload,
  ScanLine,
  X,
  FileImage,
  AlertTriangle,
  RotateCcw,
  CheckCircle2,
  Layers,
  ArrowRight,
  ShieldCheck,
} from 'lucide-react';
import { medicalApi } from '../services/api';
import { PredictionResultCard } from '../components/classifier/PredictionResultCard';
import type { SinglePredictionResponse, ModelId, ModelMetadata } from '../types';

const ALLOWED_MIME_TYPES = [
  'image/jpeg',
  'image/png',
  'image/jpg',
  'image/webp',
];

export const ImageClassifierPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<ModelId>('resnet50');
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [prediction, setPrediction] = useState<SinglePredictionResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  React.useEffect(() => {
    medicalApi.getModels().then(setModels).catch(() => {});
  }, []);

  const resnetMeta = models.find((m) => m.name === 'resnet50');
  const effnetMeta = models.find((m) => m.name === 'efficientnet_b0');

  const modelOptions: { id: ModelId; name: string; params: string; desc: string }[] = [
    {
      id: 'resnet50',
      name: 'ResNet50',
      params: resnetMeta?.total_parameters ? `${(resnetMeta.total_parameters / 1e6).toFixed(1)}M parameters` : '—',
      desc: 'Deep residual network with bottleneck blocks',
    },
    {
      id: 'efficientnet_b0',
      name: 'EfficientNetB0',
      params: effnetMeta?.total_parameters ? `${(effnetMeta.total_parameters / 1e6).toFixed(1)}M parameters` : '—',
      desc: 'Compound scaling with mobile inverted bottlenecks',
    },
  ];

  const handleFileSelect = (selectedFile: File) => {
    setErrorMessage(null);

    // Validate MIME type
    if (!ALLOWED_MIME_TYPES.includes(selectedFile.type)) {
      const ext = selectedFile.name.split('.').pop()?.toLowerCase();
      if (!['png', 'jpg', 'jpeg', 'webp'].includes(ext || '')) {
        setErrorMessage('Unsupported file format. Please upload a PNG, JPG, JPEG, or WEBP image.');
        return;
      }
    }

    // Validate 15MB limit
    if (selectedFile.size > 15 * 1024 * 1024) {
      setErrorMessage('File size exceeds 15MB limit. Please upload a smaller chest radiograph.');
      return;
    }

    setFile(selectedFile);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(URL.createObjectURL(selectedFile));
    setPrediction(null);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleReset = () => {
    setFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    setPrediction(null);
    setErrorMessage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleClassifyImage = async () => {
    if (!file) {
      setErrorMessage('Please select an image first.');
      return;
    }

    setIsPredicting(true);
    setErrorMessage(null);

    try {
      const result = await medicalApi.predictImage(file, selectedModel);
      setPrediction(result);
    } catch (err: unknown) {
      console.error('Prediction failed:', err);
      if (err && typeof err === 'object' && 'response' in err) {
        const axiosErr = err as { response?: { data?: { error?: string; detail?: string } } };
        const data = axiosErr.response?.data;
        setErrorMessage(
          data?.error || data?.detail || 'Inference failed. Check model checkpoint and image format.'
        );
      } else if (err instanceof Error) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage('Unable to connect to the prediction server. Please verify backend service.');
      }
    } finally {
      setIsPredicting(false);
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-200">
      {/* Header Banner */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight flex items-center gap-2.5">
              <ScanLine className="w-6 h-6 text-blue-600" />
              Single Image Classifier
            </h2>
            <span className="px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-blue-50 text-blue-700 border border-blue-200 tracking-wide uppercase">
              Milestone 3
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-1">
            Evaluate individual chest radiographs against fine-tuned ResNet50 or EfficientNetB0 checkpoints
          </p>
        </div>

        <div className="flex items-center gap-2 text-xs text-slate-500">
          <ShieldCheck className="w-4 h-4 text-emerald-600" />
          <span>CPU-Optimized Inference</span>
        </div>
      </div>

      {/* Prominent Medical Disclaimer Banner */}
      <div className="p-4 rounded-2xl bg-amber-50 border border-amber-200 text-amber-900 text-xs flex items-start gap-3 shadow-xs">
        <AlertTriangle className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
        <div className="space-y-0.5">
          <p className="font-bold text-amber-950">
            Educational & Research Use Only
          </p>
          <p className="text-amber-800 text-[11px] leading-relaxed">
            This model is not clinically validated and should not be used for medical diagnosis.
            Outputs represent academic algorithmic predictions on standardized chest radiographs.
          </p>
        </div>
      </div>

      {/* Error Alert Box */}
      {errorMessage && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-800 flex items-start justify-between gap-3 animate-in fade-in">
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
            <div>
              <span className="font-bold block">Inference Error</span>
              <span className="text-[11px] text-rose-700">{errorMessage}</span>
            </div>
          </div>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-rose-500 hover:text-rose-700 p-1"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Main Grid: Upload & Controls on Left, Results on Right */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-start">
        {/* Left Column: Model Selection & Image Upload */}
        <div className="space-y-5">
          {/* 1. Model Selector Card */}
          <div className="bg-white rounded-2xl p-4 sm:p-5 border border-slate-200/90 shadow-card">
            <label className="text-xs font-bold text-slate-900 uppercase tracking-wider block mb-3 flex items-center gap-2">
              <Layers className="w-4 h-4 text-blue-600" />
              Select Trained Architecture
            </label>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {modelOptions.map((m) => {
                const isSelected = selectedModel === m.id;
                return (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => {
                      setSelectedModel(m.id);
                      setPrediction(null);
                    }}
                    className={`p-3.5 rounded-xl border text-left transition-all relative min-h-[44px] ${
                      isSelected
                        ? 'border-blue-600 bg-blue-50/50 shadow-xs ring-1 ring-blue-500'
                        : 'border-slate-200 hover:border-slate-300 bg-slate-50/50 hover:bg-white'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-bold text-slate-900">
                        {m.name}
                      </span>
                      {isSelected && (
                        <CheckCircle2 className="w-4 h-4 text-blue-600 shrink-0" />
                      )}
                    </div>
                    <span className="text-[11px] font-mono text-slate-500 block">
                      {m.params}
                    </span>
                    <span className="text-[10px] text-slate-400 block mt-1 line-clamp-1">
                      {m.desc}
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* 2. Image Upload / Preview Card */}
          <div className="bg-white rounded-2xl p-4 sm:p-5 border border-slate-200/90 shadow-card">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
                Chest Radiograph Input
              </h3>
              {file && (
                <button
                  onClick={handleReset}
                  className="text-xs text-slate-500 hover:text-rose-600 flex items-center gap-1 transition-colors min-h-[36px]"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Replace Image
                </button>
              )}
            </div>

            {!file ? (
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className="border-2 border-dashed border-slate-300 hover:border-blue-500 rounded-2xl p-6 sm:p-10 text-center cursor-pointer bg-slate-50/60 hover:bg-blue-50/30 transition-all flex flex-col items-center justify-center group"
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/png,image/jpeg,image/jpg,image/webp"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      handleFileSelect(e.target.files[0]);
                    }
                  }}
                />
                <div className="w-14 h-14 rounded-2xl bg-white border border-slate-200 shadow-sm flex items-center justify-center text-slate-400 group-hover:text-blue-600 group-hover:border-blue-200 transition-colors mb-3">
                  <Upload className="w-6 h-6 stroke-[1.75]" />
                </div>
                <span className="text-sm font-semibold text-slate-800 mb-1">
                  Drag & drop radiograph here, or browse files
                </span>
                <span className="text-xs text-slate-400">
                  Supported formats: PNG, JPG, JPEG, WEBP (Max: 15MB)
                </span>
              </div>
            ) : (
              <div className="space-y-4">
                {/* Image Preview Container */}
                <div className="relative rounded-2xl overflow-hidden border border-slate-200 bg-slate-950 flex items-center justify-center min-h-[260px] max-h-[360px]">
                  {previewUrl && (
                    <img
                      src={previewUrl}
                      alt="Uploaded chest radiograph preview"
                      className="max-h-[340px] max-w-full object-contain filter contrast-110"
                    />
                  )}
                  <button
                    onClick={handleReset}
                    className="absolute top-3 right-3 p-1.5 rounded-xl bg-slate-900/80 hover:bg-rose-600 text-white transition-colors backdrop-blur-xs"
                    title="Remove image"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>

                {/* File Details bar */}
                <div className="p-3.5 rounded-xl bg-slate-50 border border-slate-200/80 flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2.5 truncate">
                    <FileImage className="w-4 h-4 text-blue-600 shrink-0" />
                    <span className="font-semibold text-slate-800 truncate">
                      {file.name}
                    </span>
                  </div>
                  <span className="font-mono text-slate-500 text-[11px] shrink-0">
                    {(file.size / 1024).toFixed(1)} KB
                  </span>
                </div>
              </div>
            )}

            {/* Action Button */}
            <div className="mt-5 pt-4 border-t border-slate-100">
              <button
                onClick={handleClassifyImage}
                disabled={!file || isPredicting}
                className="w-full py-3.5 px-4 rounded-xl text-sm font-semibold text-white bg-slate-900 hover:bg-blue-700 active:bg-blue-800 disabled:opacity-50 disabled:cursor-not-allowed shadow-md hover:shadow-glow-blue transition-all flex items-center justify-center gap-2"
              >
                {isPredicting ? (
                  <>
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    <span>Analyzing image...</span>
                  </>
                ) : (
                  <>
                    <ScanLine className="w-4 h-4" />
                    <span>Classify Image with {selectedModel === 'resnet50' ? 'ResNet50' : 'EfficientNetB0'}</span>
                    <ArrowRight className="w-4 h-4 ml-1 opacity-70" />
                  </>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Right Column: Prediction Results Card */}
        <div>
          {prediction ? (
            <PredictionResultCard prediction={prediction} />
          ) : (
            <div className="bg-white rounded-2xl p-10 border border-slate-200/90 shadow-card text-center text-slate-400 space-y-4">
              <div className="w-16 h-16 mx-auto rounded-2xl bg-slate-50 border border-slate-200 flex items-center justify-center text-slate-300">
                <ScanLine className="w-8 h-8 stroke-[1.5]" />
              </div>
              <div className="max-w-xs mx-auto">
                <h4 className="text-sm font-bold text-slate-700 mb-1">
                  Ready for Inference
                </h4>
                <p className="text-xs text-slate-500 leading-relaxed">
                  Upload a chest radiograph on the left and select your architecture, then click "Classify Image" to view verified model predictions.
                </p>
              </div>

              <div className="pt-6 border-t border-slate-100 grid grid-cols-2 gap-2 text-left text-[11px] text-slate-500">
                <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-100">
                  <span className="font-semibold block text-slate-700">ResNet50</span>
                  <span>Deep residual bottleneck network{resnetMeta?.total_parameters ? ` (${(resnetMeta.total_parameters / 1e6).toFixed(1)}M parameters)` : ''}</span>
                </div>
                <div className="p-2.5 rounded-lg bg-slate-50 border border-slate-100">
                  <span className="font-semibold block text-slate-700">EfficientNetB0</span>
                  <span>Compound scaled mobile architecture{effnetMeta?.total_parameters ? ` (${(effnetMeta.total_parameters / 1e6).toFixed(1)}M parameters)` : ''}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
