import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LabelList,
} from 'recharts';
import { BarChart3 } from 'lucide-react';
import type { ModelEvaluation } from '../../types';
import { EmptyState } from '../common/EmptyState';

interface ModelPerformanceCardProps {
  evaluations: ModelEvaluation[];
  loading?: boolean;
}

type MetricKey = 'all' | 'accuracy' | 'precision' | 'recall' | 'f1Score' | 'rocAuc';

export const ModelPerformanceCard: React.FC<ModelPerformanceCardProps> = ({
  evaluations,
  loading = false,
}) => {
  const navigate = useNavigate();
  const [selectedMetric, setSelectedMetric] = useState<MetricKey>('all');

  const resnet = evaluations.find((e) => e.modelId === 'resnet50');
  const efficientnet = evaluations.find((e) => e.modelId === 'efficientnet_b0');

  // Check if real evaluations exist
  const hasEvaluations = Boolean(
    resnet?.isTrained &&
    efficientnet?.isTrained &&
    resnet?.metrics.accuracy !== null &&
    efficientnet?.metrics.accuracy !== null
  );

  // Recharts data formatted to 2 decimals as shown in screenshot
  const allChartData = [
    {
      metric: 'Accuracy',
      key: 'accuracy',
      ResNet50: resnet?.metrics.accuracy ? Number(resnet.metrics.accuracy.toFixed(2)) : 0,
      EfficientNetB0: efficientnet?.metrics.accuracy ? Number(efficientnet.metrics.accuracy.toFixed(2)) : 0,
    },
    {
      metric: 'Precision',
      key: 'precision',
      ResNet50: resnet?.metrics.precision ? Number(resnet.metrics.precision.toFixed(2)) : 0,
      EfficientNetB0: efficientnet?.metrics.precision ? Number(efficientnet.metrics.precision.toFixed(2)) : 0,
    },
    {
      metric: 'Recall',
      key: 'recall',
      ResNet50: resnet?.metrics.recall ? Number(resnet.metrics.recall.toFixed(2)) : 0,
      EfficientNetB0: efficientnet?.metrics.recall ? Number(efficientnet.metrics.recall.toFixed(2)) : 0,
    },
    {
      metric: 'F1-Score',
      key: 'f1Score',
      ResNet50: resnet?.metrics.f1Score ? Number(resnet.metrics.f1Score.toFixed(2)) : 0,
      EfficientNetB0: efficientnet?.metrics.f1Score ? Number(efficientnet.metrics.f1Score.toFixed(2)) : 0,
    },
    {
      metric: 'AUC',
      key: 'rocAuc',
      ResNet50: resnet?.metrics.rocAuc ? Number(resnet.metrics.rocAuc.toFixed(2)) : 0,
      EfficientNetB0: efficientnet?.metrics.rocAuc ? Number(efficientnet.metrics.rocAuc.toFixed(2)) : 0,
    },
  ];

  const chartData =
    selectedMetric === 'all'
      ? allChartData
      : allChartData.filter((item) => item.key === selectedMetric);

  return (
    <div className="bg-white rounded-2xl p-5 sm:p-6 border border-slate-200/80 shadow-xs flex flex-col justify-between">
      <div>
        {/* Header and Filter Pills */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 pb-3 border-b border-slate-100">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-blue-50 border border-blue-100 flex items-center justify-center text-blue-600 shrink-0">
              <BarChart3 className="w-4 h-4" />
            </div>
            <h3 className="text-sm font-bold text-slate-900 tracking-tight">
              Model Performance Comparison
            </h3>
          </div>

          {/* Metric Selector Pills matching screenshot */}
          {hasEvaluations && (
            <div className="flex items-center gap-1 overflow-x-auto max-w-full text-xs font-medium scrollbar-none">
              {[
                { id: 'accuracy', label: 'Accuracy' },
                { id: 'precision', label: 'Precision' },
                { id: 'recall', label: 'Recall' },
                { id: 'f1Score', label: 'F1-Score' },
                { id: 'rocAuc', label: 'AUC' },
              ].map((tab) => {
                const isActive = selectedMetric === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() =>
                      setSelectedMetric(isActive ? 'all' : (tab.id as MetricKey))
                    }
                    className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold transition-all cursor-pointer whitespace-nowrap ${
                      isActive
                        ? 'bg-blue-600 text-white shadow-2xs'
                        : 'bg-slate-100 text-slate-600 hover:bg-slate-200/70 hover:text-slate-900'
                    }`}
                  >
                    {tab.label}
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Content Area */}
        {!hasEvaluations || loading ? (
          <EmptyState
            compact
            title="Evaluation not available"
            description="Initiate model training and test evaluation to view comparative metrics."
            actionText="Go to Model Training"
            onAction={() => navigate('/training')}
          />
        ) : (
          <div>
            <div className="h-64 w-full pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={chartData}
                  margin={{ top: 20, right: 10, left: -20, bottom: 0 }}
                  barGap={4}
                >
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#F1F5F9" />
                  <XAxis
                    dataKey="metric"
                    axisLine={{ stroke: '#E2E8F0' }}
                    tickLine={false}
                    tick={{ fill: '#64748B', fontSize: 11, fontWeight: 500 }}
                  />
                  <YAxis
                    domain={[0.6, 1.0]}
                    ticks={[0.6, 0.7, 0.8, 0.9, 1.0]}
                    axisLine={{ stroke: '#E2E8F0' }}
                    tickLine={false}
                    tick={{ fill: '#64748B', fontSize: 10 }}
                  />
                  <Tooltip
                    formatter={(value: any, name: any) => [`${value}`, name]}
                    contentStyle={{
                      backgroundColor: '#0F172A',
                      borderColor: '#1E293B',
                      borderRadius: '10px',
                      color: '#FFFFFF',
                      fontSize: '12px',
                    }}
                  />
                  {/* ResNet50: Blue */}
                  <Bar
                    dataKey="ResNet50"
                    fill="#3B82F6"
                    radius={[4, 4, 0, 0]}
                    name="ResNet50"
                  >
                    <LabelList
                      dataKey="ResNet50"
                      position="top"
                      fill="#3B82F6"
                      fontSize={10}
                      fontWeight={600}
                    />
                  </Bar>
                  {/* EfficientNetB0: Purple */}
                  <Bar
                    dataKey="EfficientNetB0"
                    fill="#8B5CF6"
                    radius={[4, 4, 0, 0]}
                    name="EfficientNetB0"
                  >
                    <LabelList
                      dataKey="EfficientNetB0"
                      position="top"
                      fill="#8B5CF6"
                      fontSize={10}
                      fontWeight={600}
                    />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Bottom Legend Matching Screenshot */}
            <div className="flex items-center justify-center gap-6 mt-3 pt-2 text-xs text-slate-600 font-medium">
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-[#3B82F6]" />
                <span>ResNet50</span>
              </div>
              <div className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-[#8B5CF6]" />
                <span>EfficientNetB0</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
