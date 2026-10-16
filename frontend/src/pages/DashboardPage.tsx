import React, { useEffect, useState } from 'react';
import {
  Database,
  Layers,
  Cpu,
  Award,
  AlertCircle,
} from 'lucide-react';

import { HeroSection } from '../components/dashboard/HeroSection';
import { StatCard } from '../components/common/StatCard';
import { ModelPerformanceCard } from '../components/dashboard/ModelPerformanceCard';
import { TrainingStatusCard } from '../components/dashboard/TrainingStatusCard';
import { DatasetPreviewCard } from '../components/dashboard/DatasetPreviewCard';
import { RecentActivityCard } from '../components/dashboard/RecentActivityCard';
import { QuickActionsCard } from '../components/dashboard/QuickActionsCard';
import { ProjectInfoCard } from '../components/dashboard/ProjectInfoCard';
import { CardSkeleton } from '../components/common/LoadingSkeleton';
import { medicalApi, API_BASE_URL } from '../services/api';
import type {
  DatasetInfo,
  ModelMetadata,
  ModelEvaluation,
  TrainingJob,
  ActivityEvent,
  BackendHealth,
  AnalyticsSummaryResponse,
  DatasetSample,
} from '../types';

export const DashboardPage: React.FC = () => {
  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [summary, setSummary] = useState<AnalyticsSummaryResponse | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [evaluations, setEvaluations] = useState<ModelEvaluation[]>([]);
  const [trainingJobs, setTrainingJobs] = useState<TrainingJob[]>([]);
  const [activities, setActivities] = useState<ActivityEvent[]>([]);
  const [samples, setSamples] = useState<DatasetSample[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const loadDashboardData = async () => {
    setErrorMessage(null);
    try {
      const [
        healthData,
        summaryData,
        dInfo,
        modelList,
        evals,
        jobs,
        acts,
        datasetSamples,
      ] = await Promise.all([
        medicalApi.checkHealth().catch(() => null),
        medicalApi.getAnalyticsSummary().catch(() => null),
        medicalApi.getDatasetInfo().catch(() => null),
        medicalApi.getModels().catch(() => []),
        medicalApi.getModelEvaluations().catch(() => []),
        medicalApi.getTrainingStatus().catch(() => []),
        medicalApi.getRecentActivities(15).catch(() => []),
        medicalApi.getDatasetSamples(5).catch(() => []),
      ]);

      setHealth(healthData);
      setSummary(summaryData);
      setDatasetInfo(dInfo);
      setModels(modelList);
      setEvaluations(evals);
      setTrainingJobs(jobs);
      setActivities(acts);
      setSamples(datasetSamples);
    } catch (err: unknown) {
      console.error('Failed to load dashboard metrics:', err);
      setErrorMessage(
        `Could not connect to the backend API at ${API_BASE_URL}. Please ensure the server is running.`
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadDashboardData();
  }, []);

  const isConnected = Boolean(health?.isConnected);
  const bestModel = summary?.best_model;

  return (
    <div className="space-y-6 animate-in fade-in duration-200">
      {/* Backend Disconnected Warning Banner if API is down */}
      {!isConnected && !loading && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-sm flex items-center justify-between shadow-2xs">
          <div className="flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-rose-600 shrink-0" />
            <div>
              <span className="font-bold">Backend Disconnected</span> &mdash;{' '}
              <span>
                {errorMessage ||
                  `Could not connect to FastAPI server at ${
                    health?.serverUrl || API_BASE_URL
                  }. Please verify backend is running.`}
              </span>
            </div>
          </div>
          <button
            onClick={loadDashboardData}
            className="px-3.5 py-1.5 bg-rose-100 hover:bg-rose-200 text-rose-900 rounded-lg text-xs font-semibold transition-colors shrink-0 ml-4 cursor-pointer"
          >
            Retry Connection
          </button>
        </div>
      )}

      {/* Main Grid: Left 72-75% for Research Modules, Right 25-28% for Quick Actions & Info */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* ==================== LEFT MAIN CONTENT AREA ==================== */}
        <div className="lg:col-span-8 xl:col-span-9 space-y-6">
          {/* 1. HERO SECTION */}
          <HeroSection />

          {/* 2. 4 DASHBOARD STAT CARDS */}
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
            {loading ? (
              <>
                <CardSkeleton />
                <CardSkeleton />
                <CardSkeleton />
                <CardSkeleton />
              </>
            ) : (
              <>
                {/* Stat 1: Dataset */}
                <StatCard
                  label="Dataset"
                  value={datasetInfo ? datasetInfo.name : 'PneumoniaMNIST'}
                  subValue={
                    datasetInfo?.total_samples
                      ? `${datasetInfo.total_samples.toLocaleString()} images (example)`
                      : '5,216 images (example)'
                  }
                  icon={Database}
                  iconVariant="blue"
                  linkText="View Details"
                  linkPath="/dataset"
                  thumbnailUrl={samples[0]?.imageUrl}
                />

                {/* Stat 2: Classes */}
                <StatCard
                  label="Classes"
                  value="2"
                  subValue="Normal, Pneumonia"
                  icon={Layers}
                  iconVariant="purple"
                  badge="2-"
                  linkText="View Classes"
                  linkPath="/dataset"
                />

                {/* Stat 3: Models */}
                <StatCard
                  label="Models"
                  value={models.length ? `${models.length}` : '2'}
                  subValue="ResNet50, EfficientNetB0"
                  icon={Cpu}
                  iconVariant="emerald"
                  linkText="View Models"
                  linkPath="/comparison"
                />

                {/* Stat 4: Best Model */}
                <StatCard
                  label="Best Model"
                  value={bestModel ? bestModel.display_name : 'ResNet50'}
                  subValue="(Based on validation performance)"
                  icon={Award}
                  iconVariant="amber"
                  linkText="View Comparison"
                  linkPath="/comparison"
                  isEvaluated={Boolean(bestModel)}
                />
              </>
            )}
          </div>

          {/* 3. ROW 2: Model Performance Comparison + Training Status */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ModelPerformanceCard evaluations={evaluations} loading={loading} />
            <TrainingStatusCard jobs={trainingJobs} loading={loading} />
          </div>

          {/* 4. ROW 3: Dataset Preview + Recent Activity */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <DatasetPreviewCard samples={samples} loading={loading} />
            <RecentActivityCard activities={activities} loading={loading} />
          </div>
        </div>

        {/* ==================== RIGHT SIDEBAR COLUMN ==================== */}
        <div className="lg:col-span-4 xl:col-span-3 space-y-6">
          {/* Quick Actions Card */}
          <QuickActionsCard />

          {/* Project Information Card */}
          <ProjectInfoCard
            datasetName={datasetInfo?.name || 'PneumoniaMNIST'}
            classCount={datasetInfo?.num_classes || 2}
            modelNames={
              models.length > 0
                ? models.map((m) => m.name)
                : ['ResNet50', 'EfficientNetB0']
            }
            frameworkStack="PyTorch + FastAPI + React"
          />
        </div>
      </div>
    </div>
  );
};
