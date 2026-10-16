import axios from 'axios';
import type { 
  BackendHealth, 
  DatasetInfo,
  ModelMetadata,
  ModelEvaluation, 
  TrainingJob, 
  DatasetSample, 
  RecentActivity,
  FullEvaluationReport,
  ModelComparisonResponse,
  SinglePredictionResponse,
  ModelId,
  AnalyticsSummaryResponse,
  ActivityEvent,
  PredictionRecord,
  GradCamResponse,
  ConfusionMatrixResponse,
  RocResponse,
  PrecisionRecallResponse,
  TrainingHistoryResponse,
} from '../types';

const rawBase = import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || 'http://127.0.0.1:8001';
export const API_BASE_URL = rawBase.endsWith('/api') ? rawBase : `${rawBase.replace(/\/+$/, '')}/api`;


export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 8000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to attach auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('medai_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);



export const medicalApi = {
  // Real Backend Health Check (/api/health)
  async checkHealth(): Promise<BackendHealth> {
    const start = Date.now();
    try {
      const res = await apiClient.get('/health');
      const data = res.data;
      return {
        isConnected: true,
        latencyMs: Date.now() - start,
        serverUrl: API_BASE_URL,
        service: data.service,
        version: data.version,
        device: data.device,
        cudaAvailable: data.cuda_available,
        pyTorchVersion: data.pytorch_version,
        torchvisionVersion: data.torchvision_version,
        pythonVersion: data.python_version,
        models: data.models,
      };
    } catch (err: unknown) {
      return {
        isConnected: false,
        latencyMs: Date.now() - start,
        serverUrl: API_BASE_URL,
        error: err instanceof Error ? err.message : 'Backend connection failed',
      };
    }
  },

  // Real Backend Project Information (/api/project)
  async getProjectInfo(): Promise<any> {
    const res = await apiClient.get('/project');
    return res.data;
  },

  // Real Backend Dataset Information (/api/dataset/info)
  async getDatasetInfo(): Promise<DatasetInfo> {
    const res = await apiClient.get('/dataset/info');
    return res.data;
  },

  // Real Backend Models List (/api/models)
  async getModels(): Promise<ModelMetadata[]> {
    const res = await apiClient.get('/models');
    return res.data.models;
  },

  // Real Backend Single Model Details (/api/models/{name})
  async getModel(name: string): Promise<ModelMetadata> {
    const res = await apiClient.get(`/models/${name}`);
    return res.data;
  },

  // Model Evaluation Benchmark Results (Connected to real /api/evaluation/comparison)
  async getModelEvaluations(): Promise<ModelEvaluation[]> {
    try {
      const [models, comparison] = await Promise.all([
        this.getModels(),
        this.getEvaluationComparison().catch(() => null),
      ]);

      const compMap = new Map((comparison?.models || []).map((m) => [m.model_name, m]));

      return models.map((m) => {
        const comp = compMap.get(m.name as any);
        const isEval = Boolean(comp && comp.evaluated);

        return {
          modelId: m.name as 'resnet50' | 'efficientnet_b0',
          modelName: m.display_name,
          parameters: `${(m.total_parameters / 1e6).toFixed(1)}M`,
          architecture: m.architecture,
          isTrained: isEval,
          evaluatedAt: comp?.evaluated_at || null,
          metrics: {
            accuracy: comp?.accuracy ?? null,
            precision: comp?.precision ?? null,
            recall: comp?.recall ?? null,
            sensitivity: comp?.sensitivity ?? comp?.recall ?? null,
            specificity: comp?.specificity ?? null,
            f1Score: comp?.f1 ?? null,
            rocAuc: comp?.roc_auc ?? null,
            loss: null,
            valLoss: null,
          },
        };
      });
    } catch {
      return [];
    }
  },

  // Real Test-Set Evaluation Comparison (/api/evaluation/comparison)
  async getEvaluationComparison(): Promise<ModelComparisonResponse> {
    const res = await apiClient.get('/evaluation/comparison');
    return res.data;
  },

  // Detailed Full Evaluation Report (/api/evaluation/results/{modelName})
  async getEvaluationResult(modelName: string): Promise<FullEvaluationReport> {
    const res = await apiClient.get(`/evaluation/results/${modelName}`);
    return res.data;
  },

  // Trigger real background test-set evaluation (/api/evaluation/{modelName})
  async triggerEvaluation(modelName: string): Promise<any> {
    const res = await apiClient.post(`/evaluation/${modelName}`);
    return res.data;
  },

  // Check evaluation job telemetry (/api/evaluation/status/{jobId})
  async getEvaluationStatus(jobId: string): Promise<any> {
    const res = await apiClient.get(`/evaluation/status/${jobId}`);
    return res.data;
  },


  // Real Model Training Statuses (/api/train/status)
  async getTrainingStatus(): Promise<TrainingJob[]> {
    try {
      const res = await apiClient.get('/train/status');
      return res.data.jobs.map((job: any) => ({
        id: job.job_id,
        modelId: job.model_name as 'resnet50' | 'efficientnet_b0',
        modelName: job.display_name,
        status: (job.status === 'running' ? 'training' : job.status) as any,
        currentEpoch: job.current_epoch,
        totalEpochs: job.total_epochs,
        progress: job.progress_percent,
        bestValScore: job.best_val_acc,
        bestEpoch: job.best_epoch,
        duration: job.elapsed_seconds ? `${Math.round(job.elapsed_seconds)}s` : '--',
        learningRate: job.learning_rate,
        batchSize: job.batch_size,
        optimizer: job.optimizer,
        fineTuning: job.fine_tune,
        history: job.history,
        currentTrainLoss: job.current_train_loss,
        currentValLoss: job.current_val_loss,
        checkpointSaved: job.checkpoint_saved,
        checkpointPath: job.checkpoint_path,
        errorMessage: job.error_message,
      }));
    } catch {
      return [];
    }
  },

  // Start real background training run (/api/train/start)
  async startTraining(config: {
    model_name: 'resnet50' | 'efficientnet_b0';
    epochs: number;
    batch_size: number;
    learning_rate: number;
    optimizer: 'Adam' | 'AdamW';
    fine_tune: boolean;
  }): Promise<TrainingJob> {
    const res = await apiClient.post('/train/start', config);
    const job = res.data;
    return {
      id: job.job_id,
      modelId: job.model_name as 'resnet50' | 'efficientnet_b0',
      modelName: job.display_name,
      status: (job.status === 'running' ? 'training' : job.status) as any,
      currentEpoch: job.current_epoch,
      totalEpochs: job.total_epochs,
      progress: job.progress_percent,
      bestValScore: job.best_val_acc,
      bestEpoch: job.best_epoch,
      duration: job.elapsed_seconds ? `${Math.round(job.elapsed_seconds)}s` : '--',
      learningRate: job.learning_rate,
      batchSize: job.batch_size,
      optimizer: job.optimizer,
      fineTuning: job.fine_tune,
      history: job.history,
      currentTrainLoss: job.current_train_loss,
      currentValLoss: job.current_val_loss,
      checkpointSaved: job.checkpoint_saved,
      checkpointPath: job.checkpoint_path,
      errorMessage: job.error_message,
    };
  },

  // Stop active training job (/api/train/stop/{modelId})
  async stopTraining(modelId: string): Promise<{ status: string; message: string }> {
    const res = await apiClient.post(`/train/stop/${modelId}`);
    return res.data;
  },

  // Get status of specific model training job (/api/train/status/{modelId})
  async getModelTrainingStatus(modelId: string): Promise<TrainingJob> {
    const res = await apiClient.get(`/train/status/${modelId}`);
    const job = res.data;
    return {
      id: job.job_id,
      modelId: job.model_name as 'resnet50' | 'efficientnet_b0',
      modelName: job.display_name,
      status: (job.status === 'running' ? 'training' : job.status) as any,
      currentEpoch: job.current_epoch,
      totalEpochs: job.total_epochs,
      progress: job.progress_percent,
      bestValScore: job.best_val_acc,
      bestEpoch: job.best_epoch,
      duration: job.elapsed_seconds ? `${Math.round(job.elapsed_seconds)}s` : '--',
      learningRate: job.learning_rate,
      batchSize: job.batch_size,
      optimizer: job.optimizer,
      fineTuning: job.fine_tune,
      history: job.history,
      currentTrainLoss: job.current_train_loss,
      currentValLoss: job.current_val_loss,
      checkpointSaved: job.checkpoint_saved,
      checkpointPath: job.checkpoint_path,
      errorMessage: job.error_message,
    };
  },


  // Real Dataset Samples (/api/dataset/samples)
  async getDatasetSamples(limit?: number): Promise<DatasetSample[]> {
    try {
      const url = limit ? `/dataset/samples?limit=${limit}` : '/dataset/samples';
      const res = await apiClient.get<DatasetSample[]>(url);
      if (res.data && res.data.length > 0) {
        return res.data;
      }
    } catch {
      // return empty if endpoint is temporarily unavailable
    }
    return [];
  },

  // Real Recent Activity Log (/api/activity)
  async getRecentActivity(): Promise<RecentActivity[]> {
    try {
      const events = await this.getRecentActivities(20);
      return events.map((e) => ({
        id: e.id,
        type: (e.event_type.toLowerCase().includes('train')
          ? 'training_completed'
          : e.event_type.toLowerCase().includes('pred')
          ? 'prediction_performed'
          : 'dataset_loaded') as any,
        title: e.description.split('.')[0] || e.event_type,
        description: e.description,
        timestamp: e.timestamp ? new Date(e.timestamp).toLocaleTimeString() : 'Recent',
        status: 'info' as const,
      }));
    } catch {
      return [];
    }
  },

  // Real Backend Single Image Prediction (/api/predict)
  async predictImage(file: File, modelName: ModelId = 'resnet50'): Promise<SinglePredictionResponse> {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('model_name', modelName);

    const res = await apiClient.post<SinglePredictionResponse>('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return res.data;
  },

  // Backward compatible runPrediction
  async runPrediction(formData: FormData): Promise<SinglePredictionResponse> {
    const res = await apiClient.post<SinglePredictionResponse>('/predict', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return res.data;
  },

  // Final Unified Analytics Summary (/api/analytics/summary)
  async getAnalyticsSummary(): Promise<AnalyticsSummaryResponse> {
    const res = await apiClient.get<AnalyticsSummaryResponse>('/analytics/summary');
    return res.data;
  },

  // Export JSON (/api/export/json)
  async exportJson(): Promise<Blob> {
    const res = await apiClient.get('/export/json', {
      responseType: 'blob',
    });
    return res.data;
  },

  // Export CSV (/api/export/csv)
  async exportCsv(): Promise<Blob> {
    const res = await apiClient.get('/export/csv', {
      responseType: 'blob',
    });
    return res.data;
  },

  // Export PDF Report (/api/export/report)
  async exportReportPdf(): Promise<Blob> {
    const res = await apiClient.get('/export/report', {
      responseType: 'blob',
    });
    return res.data;
  },

  // Real Research Activity Logs (/api/activity)
  async getRecentActivities(limit: number = 20): Promise<ActivityEvent[]> {
    try {
      const res = await apiClient.get<ActivityEvent[]>(`/activity?limit=${limit}`);
      return res.data;
    } catch {
      return [];
    }
  },

  // Real Prediction History Logs (/api/activity/predictions)
  async getPredictionHistory(limit: number = 20): Promise<PredictionRecord[]> {
    try {
      const res = await apiClient.get<PredictionRecord[]>(`/activity/predictions?limit=${limit}`);
      return res.data;
    } catch {
      return [];
    }
  },

  // Real Grad-CAM Generation (/api/gradcam)
  async generateGradCam(
    file: File,
    modelName: ModelId = 'resnet50',
    opacity: number = 0.65
  ): Promise<GradCamResponse> {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('model_name', modelName);
    formData.append('opacity', opacity.toString());

    const res = await apiClient.post<GradCamResponse>('/gradcam', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return res.data;
  },

  // Real Confusion Matrix Data (/api/analytics/confusion-matrix)
  async getConfusionMatrix(): Promise<ConfusionMatrixResponse> {
    const res = await apiClient.get<ConfusionMatrixResponse>('/analytics/confusion-matrix');
    return res.data;
  },

  // Real ROC Curve Data (/api/analytics/roc)
  async getRocCurve(): Promise<RocResponse> {
    const res = await apiClient.get<RocResponse>('/analytics/roc');
    return res.data;
  },

  // Real Precision-Recall Curve Data (/api/analytics/precision-recall)
  async getPrecisionRecallCurve(): Promise<PrecisionRecallResponse> {
    const res = await apiClient.get<PrecisionRecallResponse>('/analytics/precision-recall');
    return res.data;
  },

  // Real Training History Data (/api/analytics/training-history)
  async getTrainingHistory(): Promise<TrainingHistoryResponse> {
    const res = await apiClient.get<TrainingHistoryResponse>('/analytics/training-history');
    return res.data;
  },

  // Real Prediction History (/api/predictions/history)
  async getPredictionsHistory(limit: number = 20): Promise<PredictionRecord[]> {
    try {
      const res = await apiClient.get<PredictionRecord[]>(`/predictions/history?limit=${limit}`);
      return res.data;
    } catch {
      return [];
    }
  },
};

// Helper function to trigger browser file download
export function triggerBrowserDownload(blob: Blob, filename: string): void {
  const url = window.URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(anchor);
}
