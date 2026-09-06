import React from 'react';
import { Activity, SlidersHorizontal, Cpu, TrendingUp } from 'lucide-react';

interface ModelPipelineFlowProps {
  isForecasting?: boolean;
  peakStage?: string;
  peakProb?: number;
}

export const ModelPipelineFlow: React.FC<ModelPipelineFlowProps> = ({
  isForecasting = false,
  peakStage = 'Normal',
  peakProb = 0,
}) => {
  const stages = [
    {
      id: 'ingest',
      title: 'Traffic Ingested',
      badge: 'PCAP / NetFlow',
      detail: '50s sliding window (t-40s to t)',
      icon: Activity,
    },
    {
      id: 'extract',
      title: 'Feature Extraction',
      badge: '14 Telemetry Features',
      detail: 'StandardScaler normalization',
      icon: SlidersHorizontal,
    },
    {
      id: 'model',
      title: 'LSTM World Model',
      badge: 'PyTorch Rollout',
      detail: 'Hidden state autoregression',
      icon: Cpu,
    },
    {
      id: 'forecast',
      title: 'Lookahead Forecast',
      badge: `+50s | ${peakStage}`,
      detail: `${(peakProb * 100).toFixed(0)}% peak risk projection`,
      icon: TrendingUp,
    },
  ];

  return (
    <div
      id="model-pipeline-flow"
      className="bg-white rounded-lg p-4 border border-[#EAEAE5] shadow-xs"
    >
      <div className="flex items-center justify-between pb-3 border-b border-[#EAEAE5]/80 mb-3">
        <div className="flex items-center gap-2">
          <span className="text-[11px] font-semibold uppercase tracking-wider text-[#71717A]">
            Inference Pipeline Architecture
          </span>
          <span className="inline-flex items-center gap-1 font-mono text-[10px] text-[#059669] bg-emerald-50 px-2 py-0.5 rounded border border-emerald-200">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-flow-pulse" />
            Active Online Evaluation
          </span>
        </div>
        <span className="text-[11px] font-mono text-[#A1A1AA] hidden sm:inline">
          100% Client-Side Pure JS &bull; 0ms Network Latency
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 relative">
        {stages.map((stage, idx) => {
          const Icon = stage.icon;
          const isLast = idx === stages.length - 1;

          return (
            <div key={stage.id} className="relative flex flex-col justify-between">
              <div
                className={`p-3 rounded-lg border transition-all ${
                  isLast
                    ? 'bg-[#F4F4F2] border-[#D4D4D0]'
                    : 'bg-[#FAFAF8] border-[#EAEAE5]'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="w-7 h-7 rounded bg-[#18181B] text-white flex items-center justify-center shrink-0">
                    <Icon className="w-3.5 h-3.5" />
                  </div>
                  <span className="text-[10px] font-mono text-[#71717A] px-1.5 py-0.5 rounded bg-white border border-[#EAEAE5]">
                    0{idx + 1}
                  </span>
                </div>

                <div className="space-y-1">
                  <h4 className="text-xs font-semibold text-[#18181B] leading-tight">
                    {stage.title}
                  </h4>
                  <div className="font-mono text-[10px] font-medium text-[#18181B] bg-white px-1.5 py-0.5 rounded border border-[#EAEAE5] inline-block">
                    {stage.badge}
                  </div>
                  <p className="text-[11px] text-[#71717A] leading-snug">
                    {stage.detail}
                  </p>
                </div>
              </div>

              {/* Connecting Flow Indicator between stages (visible on larger screens) */}
              {!isLast && (
                <div className="hidden lg:block absolute -right-2 top-1/2 -translate-y-1/2 z-10 pointer-events-none">
                  <svg width="18" height="12" viewBox="0 0 18 12" className="overflow-visible">
                    <line
                      x1="0"
                      y1="6"
                      x2="14"
                      y2="6"
                      stroke="#18181B"
                      strokeWidth="1.5"
                      strokeDasharray="3 3"
                      className="animate-flow-dash"
                    />
                    <polygon points="12,3 17,6 12,9" fill="#18181B" />
                  </svg>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
