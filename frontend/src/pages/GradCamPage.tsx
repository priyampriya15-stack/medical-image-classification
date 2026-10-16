import React, { useState, useRef } from 'react';
import {
  Flame,
  ShieldAlert,
  Upload,
  FileImage,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Sliders,
  Sparkles,
} from 'lucide-react';
import type { ModelId, GradCamResponse } from '../types';
import { medicalApi } from '../services/api';

export const GradCamPage: React.FC = () => {
  const [selectedArchitecture, setSelectedArchitecture] = useState<ModelId>('resnet50');
  const [opacity, setOpacity] = useState<number>(65);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [gradcamResult, setGradcamResult] = useState<GradCamResponse | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Helper to load real radiograph sample from dataset
  const loadSample = async (type: 'normal' | 'pneumonia') => {
    setError(null);
    try {
      const samples = await medicalApi.getDatasetSamples();
      const targetLabel = type === 'pneumonia' ? 'Pneumonia' : 'Normal';
      const sample = samples.find(
        (s) => s.classLabel.toLowerCase() === type.toLowerCase() || s.classLabel === targetLabel
      );
      if (sample && sample.imageUrl && sample.imageUrl.startsWith('data:image')) {
        const res = await fetch(sample.imageUrl);
        const blob = await res.blob();
        const sampleFile = new File([blob], `sample_${type}.png`, { type: 'image/png' });
        setFile(sampleFile);
        setPreviewUrl(sample.imageUrl);
        setGradcamResult(null);
        return;
      }
      throw new Error('Real dataset sample not available');
    } catch {
      setError('Could not retrieve real dataset sample. Please upload a chest radiograph.');
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0];
    if (selected) {
      setFile(selected);
      setPreviewUrl(URL.createObjectURL(selected));
      setGradcamResult(null);
      setError(null);
    }
  };

  const handleGenerateGradCam = async () => {
    if (!file) {
      setError('Please upload a chest radiograph or select a sample image first.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const result = await medicalApi.generateGradCam(
        file,
        selectedArchitecture,
        opacity / 100
      );
      setGradcamResult(result);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Failed to generate Grad-CAM attention heatmap.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-200">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <span className="text-[11px] font-bold uppercase tracking-wider text-amber-600 bg-amber-50 px-2.5 py-0.5 rounded-full border border-amber-200/60">
            Interpretability & Introspection
          </span>
          <span className="text-xs font-mono text-slate-400">PneumoniaMNIST</span>
        </div>
        <h1 className="text-xl sm:text-2xl font-extrabold text-slate-900 tracking-tight flex items-center gap-2.5">
          <Flame className="w-6 h-6 text-amber-500" />
          Grad-CAM Explainability & Visual Attention
        </h1>
        <p className="text-xs text-slate-500 mt-1">
          Gradient-weighted Class Activation Mapping for inspecting convolutional feature map attention regions
        </p>
      </div>

      {/* Model Selection and Controls */}
      <div className="p-4 sm:p-5 rounded-2xl bg-white border border-slate-200/90 shadow-card flex flex-col lg:flex-row lg:items-center justify-between gap-4">
        <div className="flex flex-col sm:flex-row sm:items-center gap-2.5 w-full lg:w-auto">
          <span className="text-xs font-bold text-slate-700 uppercase tracking-wider shrink-0">
            Architecture:
          </span>
          <div className="flex flex-col sm:flex-row gap-2 w-full sm:w-auto">
            <button
              onClick={() => {
                setSelectedArchitecture('resnet50');
                setGradcamResult(null);
              }}
              className={`px-3.5 py-2.5 sm:py-1.5 rounded-xl text-xs font-semibold transition-all min-h-[44px] sm:min-h-0 flex items-center justify-center ${
                selectedArchitecture === 'resnet50'
                  ? 'bg-blue-600 text-white shadow-xs'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              ResNet50 (layer4)
            </button>
            <button
              onClick={() => {
                setSelectedArchitecture('efficientnet_b0');
                setGradcamResult(null);
              }}
              className={`px-3.5 py-2.5 sm:py-1.5 rounded-xl text-xs font-semibold transition-all min-h-[44px] sm:min-h-0 flex items-center justify-center ${
                selectedArchitecture === 'efficientnet_b0'
                  ? 'bg-cyan-600 text-white shadow-xs'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              EfficientNetB0 (features[-1])
            </button>
          </div>
        </div>

        <div className="flex items-center justify-between sm:justify-start gap-3 w-full lg:w-auto pt-2 lg:pt-0 border-t lg:border-t-0 border-slate-100">
          <div className="flex items-center gap-2">
            <Sliders className="w-4 h-4 text-slate-400 shrink-0" />
            <span className="text-xs font-semibold text-slate-600 whitespace-nowrap">
              Alpha:
            </span>
          </div>
          <div className="flex items-center gap-2 flex-1 sm:flex-initial">
            <input
              type="range"
              min="20"
              max="100"
              value={opacity}
              onChange={(e) => setOpacity(Number(e.target.value))}
              className="w-full sm:w-32 md:w-36 accent-amber-500 cursor-pointer h-5"
            />
            <span className="text-xs font-mono font-bold text-slate-700 w-9 text-right shrink-0">{opacity}%</span>
          </div>
        </div>
      </div>

      {/* Image Upload / Sample Selection Area */}
      <div className="p-4 sm:p-6 rounded-2xl bg-white border border-slate-200/90 shadow-card">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 pb-3 border-b border-slate-100">
          <div>
            <h3 className="text-sm font-bold text-slate-900">Select Input Radiograph</h3>
            <p className="text-xs text-slate-500">Upload a chest X-ray or choose a research test sample</p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => loadSample('normal')}
              className="flex-1 sm:flex-initial px-3 py-2 sm:py-1.5 text-xs font-semibold rounded-xl bg-slate-100 text-slate-700 hover:bg-slate-200 transition-colors min-h-[38px] flex items-center justify-center"
            >
              Normal Sample
            </button>
            <button
              onClick={() => loadSample('pneumonia')}
              className="flex-1 sm:flex-initial px-3 py-2 sm:py-1.5 text-xs font-semibold rounded-xl bg-amber-50 text-amber-800 border border-amber-200 hover:bg-amber-100 transition-colors min-h-[38px] flex items-center justify-center"
            >
              Pneumonia Sample
            </button>
          </div>
        </div>

        <div className="flex flex-col lg:flex-row items-stretch lg:items-center gap-4 sm:gap-6">
          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={handleFileChange}
            className="hidden"
          />

          <button
            onClick={() => fileInputRef.current?.click()}
            className="w-full lg:w-auto px-5 py-3 rounded-xl border border-dashed border-slate-300 hover:border-blue-500 bg-slate-50 hover:bg-blue-50/50 text-slate-700 flex items-center justify-center gap-2 text-xs font-semibold transition-all cursor-pointer min-h-[44px]"
          >
            <Upload className="w-4 h-4 text-blue-600 shrink-0" />
            <span className="truncate">Upload Chest Radiograph (PNG, JPG, WEBP)</span>
          </button>

          {file && (
            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 w-full lg:w-auto">
              <span className="text-xs text-slate-600 font-mono truncate max-w-xs self-center sm:self-auto">
                {file.name}
              </span>
              <button
                onClick={handleGenerateGradCam}
                disabled={loading}
                className="w-full sm:w-auto inline-flex items-center justify-center gap-2 px-5 py-2.5 text-xs font-bold text-white bg-amber-600 hover:bg-amber-500 rounded-xl transition-all shadow-xs disabled:opacity-50 min-h-[44px]"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                    <span>Computing Attention...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-3.5 h-3.5" />
                    <span>Generate Grad-CAM</span>
                  </>
                )}
              </button>
            </div>
          )}
        </div>

        {error && (
          <div className="mt-4 p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Triad Visualizer: Original Image + Heatmap + Overlay */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 sm:gap-6">
        {/* 1. Original Image */}
        <div className="bg-white rounded-2xl p-4 sm:p-5 border border-slate-200/90 shadow-card">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-100">
            <span className="text-xs font-bold text-slate-800">1. Input Radiograph</span>
            <span className="text-[10px] font-mono text-slate-400">224 x 224</span>
          </div>
          <div className="h-60 sm:h-64 rounded-xl overflow-hidden bg-slate-950 flex items-center justify-center relative">
            {gradcamResult ? (
              <img
                src={gradcamResult.original_image_base64}
                alt="Input Radiograph"
                className="w-full h-full object-contain filter contrast-110"
              />
            ) : previewUrl ? (
              <img
                src={previewUrl}
                alt="Preview Radiograph"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="text-slate-600 text-xs flex flex-col items-center gap-1.5">
                <FileImage className="w-8 h-8 text-slate-700" />
                <span>Upload an image to preview</span>
              </div>
            )}
            {gradcamResult && (
              <span className="absolute bottom-2 left-2 bg-slate-900/80 text-white text-[10px] px-2 py-0.5 rounded font-mono backdrop-blur-xs">
                {gradcamResult.predicted_class}
              </span>
            )}
          </div>
        </div>

        {/* 2. Grad-CAM Activation Heatmap */}
        <div className="bg-white rounded-2xl p-4 sm:p-5 border border-slate-200/90 shadow-card">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-100">
            <span className="text-xs font-bold text-slate-800">2. Grad-CAM Heatmap</span>
            <span className="text-[10px] font-mono text-amber-600 font-bold">Jet Colormap</span>
          </div>
          <div className="h-60 sm:h-64 rounded-xl overflow-hidden bg-slate-950 flex items-center justify-center relative">
            {gradcamResult ? (
              <img
                src={gradcamResult.heatmap_base64}
                alt="Grad-CAM Activation Heatmap"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="text-slate-600 text-xs flex flex-col items-center gap-1.5 text-center px-4">
                <Flame className="w-8 h-8 text-slate-700" />
                <span>Heatmap displays upon Grad-CAM execution</span>
              </div>
            )}
            {gradcamResult && (
              <div className="absolute bottom-2 inset-x-2 flex justify-center">
                <span className="text-[10px] font-mono text-white/90 bg-slate-900/80 px-2.5 py-0.5 rounded-full backdrop-blur-xs">
                  Target: {gradcamResult.target_layer}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* 3. Combined Overlay */}
        <div className="bg-white rounded-2xl p-4 sm:p-5 border border-slate-200/90 shadow-card">
          <div className="flex items-center justify-between mb-3 pb-2 border-b border-slate-100">
            <span className="text-xs font-bold text-slate-800">3. Attention Overlay</span>
            <span className="text-[10px] font-mono text-emerald-600 font-bold">
              {gradcamResult ? `${(gradcamResult.confidence * 100).toFixed(1)}% Conf` : 'Alpha Blended'}
            </span>
          </div>
          <div className="h-60 sm:h-64 rounded-xl overflow-hidden bg-slate-950 flex items-center justify-center relative">
            {gradcamResult ? (
              <img
                src={gradcamResult.overlay_image || gradcamResult.overlay_base64}
                alt="Explainability Overlay"
                className="w-full h-full object-contain"
              />
            ) : (
              <div className="text-slate-600 text-xs flex flex-col items-center gap-1.5 text-center px-4">
                <Sparkles className="w-8 h-8 text-slate-700" />
                <span>Blended attention overlay will render here</span>
              </div>
            )}
            {gradcamResult && (
              <div className="absolute top-2 right-2 bg-emerald-950/80 text-emerald-300 text-[10px] px-2 py-0.5 rounded font-mono border border-emerald-800/40">
                {gradcamResult.inference_time_ms} ms (CPU)
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Execution Telemetry if generated */}
      {gradcamResult && (
        <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-700 grid grid-cols-1 sm:grid-cols-2 lg:flex lg:flex-wrap items-center justify-between gap-2.5 sm:gap-4 font-mono">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
            <span className="font-bold text-slate-900">Architecture:</span>
            <span className="truncate">{gradcamResult.display_name || (gradcamResult.model_name === 'efficientnet_b0' ? 'EfficientNetB0' : 'ResNet50')}</span>
          </div>
          <div>
            <span className="font-bold text-slate-900">Target Layer:</span> {gradcamResult.target_layer}
          </div>
          <div>
            <span className="font-bold text-slate-900">Predicted Class:</span> {gradcamResult.predicted_class}
          </div>
          <div>
            <span className="font-bold text-slate-900">Confidence:</span> {(gradcamResult.confidence * 100).toFixed(2)}%
          </div>
          <div>
            <span className="font-bold text-slate-900">Inference Time:</span> {gradcamResult.inference_time_ms} ms (CPU)
          </div>
        </div>
      )}

      {/* Mandatory Academic Safety Disclaimer */}
      <div className="p-4 rounded-2xl bg-amber-50/80 border border-amber-200/90 text-xs text-amber-900 flex items-start gap-3">
        <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
        <p className="leading-relaxed">
          <span className="font-bold text-amber-950">Interpretability Guidance: </span>
          Grad-CAM highlights image regions that influenced the model prediction. It does not prove medical causation or prediction correctness.
          Model outputs are experimental research results and should not be used as a substitute for professional medical judgment.
        </p>
      </div>
    </div>
  );
};
