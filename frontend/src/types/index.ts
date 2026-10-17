export interface User {
  id: string;
  name: string;
  email: string;
  role: string;
  institution: string;
  avatarUrl?: string;
}

export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

export type ModelId = 'resnet50' | 'efficientnet_b0';

export interface DatasetSplits {
  train: number;
  validation: number;
  val?: number;
  test: number;
}

export interface DatasetInfo {
  name: string;
  description: string;
  available: boolean;
  num_classes: number;
  classes: string[];
  image_dimensions?: string;
  image_size: string;
  total_samples: number;
  splits: DatasetSplits;
  class_distribution: Record<string, number>;
  status_message?: string;
}

export interface ModelMetadata {
  name: string;
  display_name: string;
  architecture: string;
  num_classes: number;
  pretrained: boolean;
  backbone_frozen: boolean;
  checkpoint_available: boolean;
  total_parameters: number;
  trainable_parameters: number;
  checkpoint_path?: string | null;
}

export type ModelInfo = ModelMetadata;

export interface ModelMetrics {
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  sensitivity?: number | null;
  specificity?: number | null;
  f1Score: number | null;
  rocAuc: number | null;
  loss?: number | null;
  valLoss?: number | null;
}

export interface ConfusionMatrixData {
  true_negative: number;
  false_positive: number;
  false_negative: number;
  true_positive: number;
  raw_matrix: number[][];
}

export interface TestEvaluationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  sensitivity: number;
  specificity: number;
  f1: number;
  roc_auc: number;
}

export interface RocCurveData {
  fpr: number[];
  tpr: number[];
  thresholds: number[];
}

export interface PrecisionRecallCurveData {
  precision: number[];
  recall: number[];
  thresholds: number[];
}

export interface FullEvaluationReport {
  model_name: ModelId;
  display_name: string;
  dataset: string;
  split: string;
  num_samples: number;
  metrics: TestEvaluationMetrics;
  confusion_matrix: ConfusionMatrixData;
  roc_curve: RocCurveData;
  precision_recall_curve: PrecisionRecallCurveData;
  checkpoint_path: string;
  evaluated_at: string;
}

export interface ModelComparisonItem {
  model_name: ModelId;
  display_name: string;
  evaluated: boolean;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  sensitivity: number | null;
  specificity: number | null;
  f1: number | null;
  roc_auc: number | null;
  evaluated_at?: string | null;
  status_text?: string;
}

export interface ModelComparisonResponse {
  models: ModelComparisonItem[];
  primary_metric: string;
  best_model: string | null;
  comparison_summary: string | null;
}


export interface ModelEvaluation {
  modelId: ModelId;
  modelName: string;
  parameters: string;
  architecture: string;
  metrics: ModelMetrics;
  isTrained: boolean;
  evaluatedAt: string | null;
}

export type TrainingStatusType = 'idle' | 'not_started' | 'queued' | 'training' | 'running' | 'completed' | 'failed' | 'stopped';

export interface EpochMetric {
  epoch: number;
  train_loss: number;
  train_acc: number;
  val_loss: number;
  val_acc: number;
  epoch_duration_seconds: number;
}

export interface TrainingJob {
  id: string;
  modelId: ModelId;
  modelName: string;
  status: TrainingStatusType;
  currentEpoch: number;
  totalEpochs: number;
  progress: number; // 0 to 100
  bestValScore: number | null;
  bestEpoch?: number | null;
  duration: string | null; // e.g. "14m 22s" or "--"
  learningRate: number;
  batchSize: number;
  optimizer: 'Adam' | 'AdamW';
  fineTuning: boolean;
  history?: EpochMetric[];
  currentTrainLoss?: number | null;
  currentValLoss?: number | null;
  checkpointSaved?: boolean;
  checkpointPath?: string | null;
  startedAt?: string | null;
  completedAt?: string | null;
  errorMessage?: string;
}

export interface TrainingConfigRequest {
  model_name: ModelId;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  optimizer: 'Adam' | 'AdamW';
  fine_tune: boolean;
}


export interface DatasetSample {
  id: string;
  imageUrl: string;
  classLabel: 'Normal' | 'Pneumonia';
  split: 'train' | 'val' | 'test';
  dimensions: string;
}

export type ActivityType = 
  | 'dataset_loaded'
  | 'training_started'
  | 'training_completed'
  | 'comparison_completed'
  | 'prediction_performed';

export interface RecentActivity {
  id: string;
  type: ActivityType;
  title: string;
  description: string;
  timestamp: string;
  status: 'info' | 'success' | 'warning' | 'error';
}

export interface BackendHealth {
  isConnected: boolean;
  latencyMs: number;
  serverUrl: string;
  service?: string;
  version?: string;
  device?: string; // e.g. "CUDA (NVIDIA RTX)" or "CPU"
  cudaAvailable?: boolean;
  pyTorchVersion?: string;
  torchvisionVersion?: string;
  pythonVersion?: string;
  models?: Record<string, boolean>;
  error?: string;
}

export interface PredictionScore {
  className: string;
  probability: number;
}

export interface PredictionOutput {
  id: string;
  imageName: string;
  previewUrl: string;
  predictedClass: 'Normal' | 'Pneumonia';
  confidence: number;
  classScores: {
    Normal: number;
    Pneumonia: number;
  };
  resnet50?: {
    predictedClass: 'Normal' | 'Pneumonia';
    confidence: number;
  };
  efficientnetB0?: {
    predictedClass: 'Normal' | 'Pneumonia';
    confidence: number;
  };
  inferenceTimeMs: number;
  timestamp: string;
}

export interface ModelPredictionInfo {
  name: ModelId;
  display_name: string;
}

export interface PredictionResultData {
  class: 'Normal' | 'Pneumonia';
  class_index: number;
  confidence: number;
}

export interface ClassProbabilitiesData {
  Normal: number;
  Pneumonia: number;
}

export interface ImageMetadata {
  filename: string;
  format: string;
  width: number;
  height: number;
}

export interface SinglePredictionResponse {
  success: boolean;
  model: ModelPredictionInfo;
  prediction: PredictionResultData;
  probabilities: ClassProbabilitiesData;
  image: ImageMetadata;
  inference_time_ms: number;
  device: string;
  model_name?: string;
  predicted_class?: string;
  class_index?: number;
  confidence?: number;
}

export interface ProjectInfo {
  name: string;
  dataset: string;
  num_classes: number;
  classes: string[];
}

export interface BestModelInfo {
  name: string;
  display_name: string;
  primary_metric: string;
  value: number;
}

export interface ModelAnalyticsSummaryItem {
  name: ModelId;
  display_name: string;
  training_status: string;
  evaluation_status: string;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  sensitivity: number | null;
  specificity: number | null;
  f1: number | null;
  roc_auc: number | null;
  total_parameters?: number | null;
  trainable_parameters?: number | null;
  test_samples?: number | null;
  evaluated_at?: string | null;
  confusion_matrix?: ConfusionMatrixData | null;
}

export interface AnalyticsSummaryResponse {
  project: ProjectInfo;
  models: ModelAnalyticsSummaryItem[];
  best_model: BestModelInfo | null;
  research_interpretation?: string | null;
  generated_at: string;
  disclaimer: string;
}

export interface ActivityEvent {
  id: string;
  timestamp: string;
  event_type: 'TRAINING_COMPLETED' | 'EVALUATION_COMPLETED' | 'PREDICTION_COMPLETED' | 'GRADCAM_GENERATED' | string;
  model_name: string;
  description: string;
  metadata?: Record<string, any>;
}

export interface PredictionRecord {
  id: string;
  timestamp: string;
  model_name: string;
  predicted_class: string;
  confidence: number;
  inference_time_ms: number;
}

export interface GradCamResponse {
  success: boolean;
  model_name: ModelId;
  display_name: string;
  target_layer: string;
  predicted_class: 'Normal' | 'Pneumonia';
  class_index: number;
  confidence: number;
  original_image_base64: string;
  heatmap_base64: string;
  overlay_base64: string;
  overlay_image?: string;
  heatmap_available?: boolean;
  inference_time_ms: number;
  device: string;
}

export interface ModelConfusionMatrixItem {
  matrix: number[][];
  raw_matrix?: number[][];
  true_negative: number;
  false_positive: number;
  false_negative: number;
  true_positive: number;
}

export interface ConfusionMatrixResponse {
  classes: string[];
  resnet50?: ModelConfusionMatrixItem | null;
  efficientnet_b0?: ModelConfusionMatrixItem | null;
}

export interface ModelRocData {
  false_positive_rate: number[];
  true_positive_rate: number[];
  fpr?: number[];
  tpr?: number[];
  thresholds?: number[];
  roc_auc: number;
}

export interface RocResponse {
  resnet50?: ModelRocData | null;
  efficientnet_b0?: ModelRocData | null;
}

export interface ModelPrecisionRecallData {
  precision: number[];
  recall: number[];
  thresholds?: number[];
  average_precision: number | null;
}

export interface PrecisionRecallResponse {
  resnet50?: ModelPrecisionRecallData | null;
  efficientnet_b0?: ModelPrecisionRecallData | null;
}

export interface TrainingHistoryEpoch {
  epoch: number;
  train_loss: number;
  validation_loss: number;
  train_accuracy: number;
  validation_accuracy: number;
  train_acc?: number;
  val_acc?: number;
  learning_rate: number;
  epoch_duration_seconds?: number;
}

export interface TrainingHistoryResponse {
  resnet50?: TrainingHistoryEpoch[] | null;
  efficientnet_b0?: TrainingHistoryEpoch[] | null;
}

