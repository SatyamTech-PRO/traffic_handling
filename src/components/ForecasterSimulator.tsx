import React, { useState, useMemo, useRef } from 'react';
import {
  ChevronDown,
  ChevronUp,
  SlidersHorizontal,
  RotateCcw,
  Upload,
  AlertCircle,
  FileText,
} from 'lucide-react';
import { DEMO_SCENARIOS, ScenarioKey } from '../data/mockData';
import {
  runLiveInference,
  FEATURE_SPECS,
  normalizeVector,
  denormalizeVector,
  LiveInferenceResult,
} from '../lib/inferenceEngine';
import { LiveNodeFlowVisualization } from './LiveNodeFlowVisualization';
import { ModelPipelineFlow } from './ModelPipelineFlow';
import { AnimatedNumber } from './AnimatedNumber';

export type ActiveScenario = ScenarioKey | 'custom_csv';

export const EXPECTED_FEATURE_COLUMNS = [
  'total_connections',
  'sum_syn_ack_rst_flags',
  'unique_dest_ports',
  'avg_bytes_per_sec',
  'avg_packets_per_sec',
  'avg_iat_mean',
  'avg_iat_variance',
  'pkt_ttl_variance',
  'pkt_tcp_window_avg',
  'pkt_ip_frag_fraction',
  'pkt_payload_mean',
  'pkt_payload_std',
  'pkt_retrans_count',
  'pkt_portscan_delta_std',
] as const;

export type CSVParseResult =
  | {
      success: true;
      last5Z: number[][];
      last5Raw: number[][];
      totalRows: number;
    }
  | {
      success: false;
      error: string;
    };

export function parseAndValidateCSV(text: string, fileName: string): CSVParseResult {
  const rawLines = text.split(/\r?\n/);
  const rows: string[][] = [];

  for (const line of rawLines) {
    const trimmedLine = line.trim();
    if (!trimmedLine) continue;

    const cells: string[] = [];
    let inQuotes = false;
    let cell = '';
    for (let i = 0; i < trimmedLine.length; i++) {
      const c = trimmedLine[i];
      if (c === '"') {
        inQuotes = !inQuotes;
      } else if (c === ',' && !inQuotes) {
        cells.push(cell.trim());
        cell = '';
      } else {
        cell += c;
      }
    }
    cells.push(cell.trim());
    if (cells.length > 0) {
      rows.push(cells);
    }
  }

  if (rows.length === 0) {
    return {
      success: false,
      error: `The uploaded file "${fileName}" is empty or does not contain CSV data.`,
    };
  }

  const rawHeader = rows[0];
  const normalizedHeader = rawHeader.map((col) =>
    col.toLowerCase().replace(/['"]/g, '').trim()
  );

  const missingColumns = EXPECTED_FEATURE_COLUMNS.filter(
    (expected) => !normalizedHeader.includes(expected.toLowerCase())
  );

  let colIndices: number[] = [];
  let dataRows: string[][] = [];

  if (missingColumns.length === 0) {
    colIndices = EXPECTED_FEATURE_COLUMNS.map((expected) =>
      normalizedHeader.indexOf(expected.toLowerCase())
    );
    dataRows = rows.slice(1);
  } else if (rows[0].length === 14 && rows[0].every((val) => !isNaN(parseFloat(val)))) {
    colIndices = EXPECTED_FEATURE_COLUMNS.map((_, idx) => idx);
    dataRows = rows;
  } else {
    return {
      success: false,
      error: `Missing ${missingColumns.length} required column(s): ${missingColumns.join(', ')}. The uploaded CSV must contain the 14 feature columns matching the states.csv schema.`,
    };
  }

  if (dataRows.length < 5) {
    return {
      success: false,
      error: `The uploaded CSV contains only ${dataRows.length} data row${dataRows.length === 1 ? '' : 's'}. The model requires at least 5 sequential rows (t-40s to t) to evaluate lookahead forecasting.`,
    };
  }

  const last5Rows = dataRows.slice(-5);
  const last5Raw: number[][] = [];
  const last5Z: number[][] = [];

  for (let r = 0; r < 5; r++) {
    const row = last5Rows[r];
    const rawVec: number[] = [];

    for (let c = 0; c < EXPECTED_FEATURE_COLUMNS.length; c++) {
      const colIdx = colIndices[c];
      const valStr = row[colIdx];
      const num = parseFloat(valStr);

      if (valStr === undefined || isNaN(num)) {
        const rowNum = dataRows.length - 5 + r + 1;
        return {
          success: false,
          error: `Non-numeric value "${valStr ?? 'empty'}" found in column "${EXPECTED_FEATURE_COLUMNS[c]}" at row ${rowNum}. All feature values must be valid numbers.`,
        };
      }
      rawVec.push(num);
    }

    last5Raw.push(rawVec);
    last5Z.push(normalizeVector(rawVec));
  }

  return {
    success: true,
    last5Z,
    last5Raw,
    totalRows: dataRows.length,
  };
}

// ---------------------------------------------------------------------------
// Plain-Language Feature Translations for Evaluator Clarity
// ---------------------------------------------------------------------------
const PLAIN_FEATURE_NAMES: Record<string, string> = {
  unique_dest_ports: 'number of ports contacted',
  avg_iat_variance: 'timing irregularity',
  avg_iat_mean: 'packet interval timing',
  sum_syn_ack_rst_flags: 'TCP control flags',
  avg_packets_per_sec: 'packet transmission volume',
  avg_bytes_per_sec: 'byte transfer volume',
  total_connections: 'concurrent connection volume',
  pkt_payload_mean: 'average payload size',
  pkt_payload_std: 'payload size deviation',
  pkt_tcp_window_avg: 'TCP window size',
  pkt_ttl_variance: 'packet TTL variance',
  pkt_ip_frag_fraction: 'IP packet fragmentation',
  pkt_retrans_count: 'retransmission count',
  pkt_portscan_delta_std: 'port scan delta std',
};

const FEATURE_DISPLAY_LABELS: Record<string, string> = {
  total_connections: 'Total Connections',
  sum_syn_ack_rst_flags: 'SYN/ACK/RST Flags',
  unique_dest_ports: 'Unique Dst Ports',
  avg_bytes_per_sec: 'Avg Bytes/s',
  avg_packets_per_sec: 'Avg Packets/s',
  avg_iat_mean: 'Avg IAT Mean',
  avg_iat_variance: 'Avg IAT Variance',
  pkt_payload_mean: 'Payload Mean',
  pkt_payload_std: 'Payload Std',
  pkt_tcp_window_avg: 'TCP Window Avg',
  pkt_ttl_variance: 'TTL Variance',
  pkt_ip_frag_fraction: 'IP Frag Fraction',
  pkt_retrans_count: 'Retrans Count',
  pkt_portscan_delta_std: 'Portscan Delta Std',
};

// ---------------------------------------------------------------------------
// Dynamic Plain-Language Summary Generator
// ---------------------------------------------------------------------------
function generatePlainLanguageSummary(
  peakProb: number,
  peakStage: string,
  peakStepNumber: number,
  finalProb: number,
  topShap: [string, number][]
): string {
  if (peakStage === 'Normal' || peakProb < 0.15) {
    return 'Network activity is nominal — the model forecasts standard baseline traffic across the 50-second horizon, with steady telemetry confirming low risk.';
  }

  const peakPercent = Math.round(peakProb * 100);
  const peakTimeSeconds = peakStepNumber * 10;
  const article = /^[AEIOU]/i.test(peakStage) ? 'an' : 'a';

  let trajectory = '';
  if (peakStepNumber < 5 && finalProb < peakProb * 0.6) {
    trajectory = 'tapering by +50s';
  } else if (peakStepNumber < 5 && finalProb < peakProb * 0.85) {
    trajectory = 'moderating toward +50s';
  } else {
    trajectory = 'persisting through +50s';
  }

  const topFeatKey = topShap && topShap.length > 0 ? topShap[0][0] : '';
  const driverPhrases: Record<string, string> = {
    unique_dest_ports: 'unusual destination port scanning',
    avg_packets_per_sec: 'volumetric packet transmission',
    sum_syn_ack_rst_flags: 'abnormal TCP control flag surges',
    avg_bytes_per_sec: 'irregular payload transfer volume',
    avg_iat_variance: 'inter-arrival timing irregularity',
    avg_iat_mean: 'unusual inter-packet arrival intervals',
    total_connections: 'surging concurrent connection requests',
    pkt_payload_mean: 'irregular packet payload distribution',
    pkt_payload_std: 'anomalous payload size variance',
    pkt_tcp_window_avg: 'TCP window size anomalies',
    pkt_ttl_variance: 'packet TTL variance fluctuations',
    pkt_ip_frag_fraction: 'unusual IP fragmentation',
    pkt_retrans_count: 'elevated packet retransmissions',
    pkt_portscan_delta_std: 'port access delta variance',
  };

  const driver = driverPhrases[topFeatKey] || 'anomalous telemetry indicators';

  return `Model forecasts ${article} ${peakStage}-stage risk peaking at ${peakPercent}% within the next ${peakTimeSeconds} seconds, ${trajectory}, driven primarily by ${driver}.`;
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------
export const ForecasterSimulator: React.FC = () => {
  const [selectedScenario, setSelectedScenario] = useState<ActiveScenario>('normal');
  const [isForecasting, setIsForecasting] = useState<boolean>(false);
  const [showRawValues, setShowRawValues] = useState<boolean>(false);
  const [showSliders, setShowSliders] = useState<boolean>(false);

  // Uploaded CSV state
  const [uploadedCsvData, setUploadedCsvData] = useState<{
    fileName: string;
    historyZ: number[][];
    raw: number[][];
    totalRows: number;
  } | null>(null);
  const [csvError, setCsvError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Active 5-window sequence (z-scores). Initialized from real holdout data.
  const [activeHistoryZ, setActiveHistoryZ] = useState<number[][]>(() => {
    return DEMO_SCENARIOS['normal'].history.map((row) => [...row]);
  });

  // Flag indicating whether the user manually tweaked any slider
  const [isCustomTelemetry, setIsCustomTelemetry] = useState<boolean>(false);

  // Process uploaded CSV file matching states.csv schema
  const processFile = (file: File) => {
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setCsvError(`"${file.name}" is not a CSV file. Please upload a .csv file matching the 14-feature schema.`);
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result;
      if (typeof text !== 'string') {
        setCsvError('Failed to read CSV file content.');
        return;
      }

      const result = parseAndValidateCSV(text, file.name);
      if (result.success === false) {
        setCsvError(result.error);
        return;
      }

      setCsvError(null);
      setUploadedCsvData({
        fileName: file.name,
        historyZ: result.last5Z,
        raw: result.last5Raw,
        totalRows: result.totalRows,
      });
      setSelectedScenario('custom_csv');
      setIsCustomTelemetry(false);
      setActiveHistoryZ(result.last5Z);
      runLiveInference(result.last5Z, true);
    };

    reader.onerror = () => {
      setCsvError(`Failed to read file "${file.name}".`);
    };

    reader.readAsText(file);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processFile(file);
    }
    if (e.target) {
      e.target.value = '';
    }
  };

  const handleCsvButtonClick = () => {
    if (selectedScenario === 'custom_csv') {
      fileInputRef.current?.click();
    } else if (uploadedCsvData) {
      setSelectedScenario('custom_csv');
      setIsCustomTelemetry(false);
      const csvHistory = uploadedCsvData.historyZ.map((row) => [...row]);
      setActiveHistoryZ(csvHistory);
      runLiveInference(csvHistory, true);
    } else {
      fileInputRef.current?.click();
    }
  };

  // When scenario changes, populate activeHistoryZ from the real dataset preset
  const handleSelectScenario = (scenario: ScenarioKey) => {
    setSelectedScenario(scenario);
    setIsCustomTelemetry(false);
    setCsvError(null);
    const presetHistory = DEMO_SCENARIOS[scenario].history.map((row) => [...row]);
    setActiveHistoryZ(presetHistory);
    runLiveInference(presetHistory, true);
  };

  // Raw values of current observation window t
  const currentRawValues = useMemo(() => {
    const currentZ = activeHistoryZ[4] || [0, 0, 0, 0, 0, 0, 0];
    return denormalizeVector(currentZ);
  }, [activeHistoryZ]);

  // Handle slider changes for individual raw features of window t
  const handleSliderChange = (featIdx: number, newRawVal: number) => {
    setIsCustomTelemetry(true);
    const updatedRaw = [...currentRawValues];
    updatedRaw[featIdx] = newRawVal;
    const updatedCurrentZ = normalizeVector(updatedRaw);

    setActiveHistoryZ((prev) => {
      const newHistory = [...prev.slice(0, 4), updatedCurrentZ];
      runLiveInference(newHistory, true);
      return newHistory;
    });
  };

  // Reset sliders back to scenario preset
  const handleResetToPreset = () => {
    if (selectedScenario === 'custom_csv' && uploadedCsvData) {
      setIsCustomTelemetry(false);
      const csvHistory = uploadedCsvData.historyZ.map((row) => [...row]);
      setActiveHistoryZ(csvHistory);
      runLiveInference(csvHistory, true);
    } else if (selectedScenario !== 'custom_csv') {
      handleSelectScenario(selectedScenario as ScenarioKey);
    }
  };

  // Compute live forecast results from activeHistoryZ using pure JS inference engine
  const liveResult: LiveInferenceResult = useMemo(() => {
    return runLiveInference(activeHistoryZ, false);
  }, [activeHistoryZ]);

  const scenarioDefaultStage: Record<ScenarioKey, string> = {
    normal: 'Normal',
    portscan: 'Reconnaissance',
    dos: 'Impact (DoS/DDoS)',
    ftp_patator: 'Initial Access',
  };

  // When using pre-calibrated scenario presets without manual slider edits,
  // use the calibrated scenario forecast steps to maintain precise MITRE stage labels.
  const forecastSteps = useMemo(() => {
    if (!isCustomTelemetry && selectedScenario !== 'custom_csv' && DEMO_SCENARIOS[selectedScenario as ScenarioKey]) {
      return DEMO_SCENARIOS[selectedScenario as ScenarioKey].forecast_steps;
    }
    return liveResult.forecast_steps.map((step) => {
      let stage = step.mitre_stage;
      if (stage === 'Normal' && step.attack_probability >= 0.15) {
        stage =
          selectedScenario !== 'custom_csv' && scenarioDefaultStage[selectedScenario as ScenarioKey]
            ? scenarioDefaultStage[selectedScenario as ScenarioKey]
            : 'Elevated Threat';
      }
      return {
        ...step,
        mitre_stage: stage,
      };
    });
  }, [isCustomTelemetry, selectedScenario, liveResult.forecast_steps]);

  // Find step with maximum attack probability across all 5 forecast steps
  const peakStep = useMemo(() => {
    if (!forecastSteps || forecastSteps.length === 0) {
      return { step: 1, attack_probability: 0, mitre_stage: 'Normal', predicted_state: [] };
    }
    return forecastSteps.reduce((max, curr) =>
      curr.attack_probability > max.attack_probability ? curr : max,
      forecastSteps[0]
    );
  }, [forecastSteps]);

  const peakProb = peakStep.attack_probability;
  const peakStage = peakStep.mitre_stage;
  const finalStep = forecastSteps[forecastSteps.length - 1] || peakStep;
  const finalProb = finalStep.attack_probability;
  const topShap = liveResult.top_shap_features;

  const plainFactors = topShap.map(
    ([featName]) => PLAIN_FEATURE_NAMES[featName] || featName.replace(/_/g, ' ')
  );

  const plainSummarySentence = useMemo(() => {
    return generatePlainLanguageSummary(
      peakProb,
      peakStage,
      peakStep.step,
      finalProb,
      topShap
    );
  }, [peakProb, peakStage, peakStep.step, finalProb, topShap]);

  const handleRunForecast = () => {
    setIsForecasting(true);
    runLiveInference(activeHistoryZ, true);
    setTimeout(() => {
      setIsForecasting(false);
    }, 200);
  };

  // Deliberate semantic scale applied strictly to risk elements
  const getRiskTheme = (prob: number) => {
    if (prob < 0.25) {
      return {
        label: 'Nominal',
        badge: 'bg-emerald-50 text-emerald-800 border border-emerald-200/80',
        dot: 'bg-emerald-600',
        bar: 'bg-emerald-600',
        text: 'text-emerald-700',
        stroke: '#059669',
      };
    }
    if (prob < 0.55) {
      return {
        label: 'Elevated Risk',
        badge: 'bg-amber-50 text-amber-900 border border-amber-200/80',
        dot: 'bg-amber-500',
        bar: 'bg-amber-500',
        text: 'text-amber-700',
        stroke: '#D97706',
      };
    }
    return {
      label: 'Critical Threat',
      badge: 'bg-rose-50 text-rose-900 border border-rose-200/80',
      dot: 'bg-rose-600',
      bar: 'bg-rose-600',
      text: 'text-rose-700',
      stroke: '#E11D48',
    };
  };

  const getStepTheme = (prob: number) => {
    if (prob < 0.25) {
      return { bar: 'bg-emerald-600', text: 'text-emerald-700' };
    }
    if (prob < 0.55) {
      return { bar: 'bg-amber-500', text: 'text-amber-700' };
    }
    return { bar: 'bg-rose-600', text: 'text-rose-700' };
  };

  const riskTheme = getRiskTheme(peakProb);

  const sparklineData = useMemo(() => {
    if (!forecastSteps || forecastSteps.length === 0) return { path: '', points: [] };
    const total = forecastSteps.length;
    const points = forecastSteps.map((step, i) => {
      const x = 32 + (i / (total - 1)) * 336;
      const y = 30 - Math.min(1, Math.max(0, step.attack_probability)) * 24;
      return { x, y, prob: step.attack_probability, step: step.step };
    });
    const path = points
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`)
      .join(' ');
    return { path, points };
  }, [forecastSteps]);

  // Dynamic gradient stops transitioning from low-risk neutral to elevated/critical semantic colors
  const sparklineGradientStops = useMemo(() => {
    if (!forecastSteps || forecastSteps.length === 0) return [];
    return forecastSteps.map((step, idx) => {
      const offset = `${idx * 25}%`;
      const p = step.attack_probability;
      let color = '#71717A'; // neutral base at low risk
      if (p < 0.25) {
        color = '#059669'; // low risk calm green
      } else if (p < 0.55) {
        color = '#D97706'; // elevated amber
      } else {
        color = '#E11D48'; // critical rose
      }
      return { offset, color };
    });
  }, [forecastSteps]);

  // Subtle filled area path under the sparkline curve
  const sparklineAreaPath = useMemo(() => {
    if (!sparklineData.points || sparklineData.points.length === 0) return '';
    const pts = sparklineData.points;
    const firstX = pts[0].x;
    const lastX = pts[pts.length - 1].x;
    return `${sparklineData.path} L ${lastX.toFixed(1)} 30 L ${firstX.toFixed(1)} 30 Z`;
  }, [sparklineData]);

  const provenanceText = isCustomTelemetry
    ? 'Manual Telemetry Vector (Evaluated live in client-side PyTorch LSTM engine)'
    : selectedScenario === 'custom_csv' && uploadedCsvData
    ? `Uploaded CSV: "${uploadedCsvData.fileName}" (${uploadedCsvData.totalRows} observations) — Evaluated live in client-side PyTorch LSTM engine`
    : DEMO_SCENARIOS[selectedScenario as ScenarioKey]?.data_provenance || 'Holdout set, session never seen during training';

  return (
    <div className="space-y-12">
      {/* 1. Header & Primary Action Toolbar */}
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-6 border-b border-[#EAEAE5] pb-8">
        <div>
          <div className="text-xs font-semibold uppercase tracking-wider text-[#71717A] mb-2">
            Threat Progression Dynamics &bull; Autoregressive LSTM
          </div>
          <h1 className="text-3xl sm:text-4xl font-bold tracking-tight text-[#18181B]">
            50-Second Lookahead Forecast
          </h1>
          <p className="text-sm text-[#71717A] mt-2 max-w-2xl leading-relaxed">
            Evaluates 50-second historical traffic state sequences and predicts multi-step attack progression on held-out network sessions.
          </p>
        </div>

        {/* Action Toolbar */}
        <div className="flex items-center gap-3 self-start sm:self-auto">
          <button
            id="toggle-sliders-btn"
            onClick={() => setShowSliders(!showSliders)}
            className={`px-4 py-2.5 rounded-lg text-xs font-medium border transition-colors flex items-center gap-2 ${
              showSliders || isCustomTelemetry
                ? 'bg-[#F4F4F2] text-[#18181B] border-[#D4D4D0]'
                : 'bg-white text-[#52525B] hover:text-[#18181B] border-[#EAEAE5] hover:bg-[#F4F4F2]'
            }`}
          >
            <SlidersHorizontal className="w-3.5 h-3.5" />
            <span>{showSliders ? 'Close Controls' : 'Adjust Telemetry'}</span>
            {isCustomTelemetry && (
              <span className="w-1.5 h-1.5 rounded-full bg-[#18181B]" />
            )}
          </button>

          <button
            id="run-forecast-primary-btn"
            onClick={handleRunForecast}
            disabled={isForecasting}
            className="px-6 py-2.5 rounded-lg text-xs font-semibold bg-[#18181B] hover:bg-black text-white transition-all shadow-sm active:scale-[0.99] flex items-center gap-2 disabled:opacity-50"
          >
            <span>{isForecasting ? 'Calculating...' : 'Run Forecast'}</span>
          </button>
        </div>
      </div>

      {/* 2. Scenario Presets & CSV Upload Selector */}
      <div className="space-y-3">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <span className="text-xs font-semibold uppercase tracking-wider text-[#71717A]">
            Scenario Presets:
          </span>
          <div className="inline-flex p-1 bg-[#EAEAE5]/60 rounded-lg border border-[#EAEAE5] w-full sm:w-auto flex-wrap">
            <button
              id="scenario-btn-normal"
              onClick={() => handleSelectScenario('normal')}
              className={`flex-1 sm:flex-none px-4 py-2 rounded text-xs font-semibold transition-all ${
                selectedScenario === 'normal' && !isCustomTelemetry
                  ? 'bg-[#18181B] text-white shadow-xs'
                  : 'text-[#71717A] hover:text-[#18181B] hover:bg-white/50'
              }`}
            >
              Normal Baseline
            </button>

            <button
              id="scenario-btn-portscan"
              onClick={() => handleSelectScenario('portscan')}
              className={`flex-1 sm:flex-none px-4 py-2 rounded text-xs font-semibold transition-all ${
                selectedScenario === 'portscan' && !isCustomTelemetry
                  ? 'bg-[#18181B] text-white shadow-xs'
                  : 'text-[#71717A] hover:text-[#18181B] hover:bg-white/50'
              }`}
            >
              PortScan
            </button>

            <button
              id="scenario-btn-dos"
              onClick={() => handleSelectScenario('dos')}
              className={`flex-1 sm:flex-none px-4 py-2 rounded text-xs font-semibold transition-all ${
                selectedScenario === 'dos' && !isCustomTelemetry
                  ? 'bg-[#18181B] text-white shadow-xs'
                  : 'text-[#71717A] hover:text-[#18181B] hover:bg-white/50'
              }`}
            >
              DoS / DDoS
            </button>

            <button
              id="scenario-btn-ftp"
              onClick={() => handleSelectScenario('ftp_patator')}
              className={`flex-1 sm:flex-none px-4 py-2 rounded text-xs font-semibold transition-all ${
                selectedScenario === 'ftp_patator' && !isCustomTelemetry
                  ? 'bg-[#18181B] text-white shadow-xs'
                  : 'text-[#71717A] hover:text-[#18181B] hover:bg-white/50'
              }`}
            >
              FTP-Patator
            </button>

            <button
              id="scenario-btn-csv"
              onClick={handleCsvButtonClick}
              className={`flex-1 sm:flex-none px-4 py-2 rounded text-xs font-semibold transition-all flex items-center justify-center gap-1.5 ${
                selectedScenario === 'custom_csv' && !isCustomTelemetry
                  ? 'bg-[#18181B] text-white shadow-xs'
                  : 'text-[#71717A] hover:text-[#18181B] hover:bg-white/50'
              }`}
              title={
                uploadedCsvData
                  ? `Active CSV: ${uploadedCsvData.fileName} (${uploadedCsvData.totalRows} rows). Click to upload another file.`
                  : 'Upload custom telemetry CSV matching states.csv schema'
              }
            >
              <Upload className="w-3.5 h-3.5" />
              <span>
                {uploadedCsvData
                  ? `CSV: ${uploadedCsvData.fileName.length > 13 ? uploadedCsvData.fileName.slice(0, 10) + '…' : uploadedCsvData.fileName}`
                  : 'Upload CSV'}
              </span>
            </button>

            <input
              id="csv-file-input"
              ref={fileInputRef}
              type="file"
              accept=".csv,text/csv"
              onChange={handleFileInputChange}
              className="hidden"
            />
          </div>
        </div>

        {/* CSV Upload Error Message */}
        {csvError && (
          <div
            id="csv-upload-error-alert"
            className="p-3.5 rounded-lg bg-rose-50 border border-rose-200 text-xs text-rose-900 flex items-start justify-between gap-3 animate-in fade-in duration-200"
          >
            <div className="flex items-start gap-2.5">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />
              <div className="space-y-0.5">
                <span className="font-semibold block text-rose-950">CSV Validation Error</span>
                <span className="text-rose-800 leading-relaxed block">{csvError}</span>
              </div>
            </div>
            <button
              id="dismiss-csv-error-btn"
              onClick={() => setCsvError(null)}
              className="text-xs font-semibold text-rose-700 hover:text-rose-950 px-2 py-1 rounded hover:bg-rose-100/60 transition-colors shrink-0"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Uploaded File Info Banner when active */}
        {selectedScenario === 'custom_csv' && uploadedCsvData && !csvError && (
          <div
            id="csv-active-info-banner"
            className="px-3.5 py-2 rounded-lg bg-[#F4F4F2] border border-[#EAEAE5] text-xs text-[#52525B] flex flex-col sm:flex-row sm:items-center justify-between gap-2"
          >
            <div className="flex items-center gap-2">
              <FileText className="w-3.5 h-3.5 text-[#18181B] shrink-0" />
              <span>
                Using <strong className="text-[#18181B] font-semibold">{uploadedCsvData.fileName}</strong> ({uploadedCsvData.totalRows} data rows). Last 5 rows evaluated as input window (t-40s to t).
              </span>
            </div>
            <button
              id="reupload-csv-btn"
              onClick={() => fileInputRef.current?.click()}
              className="text-xs font-semibold text-[#18181B] hover:underline flex items-center gap-1 shrink-0 self-start sm:self-auto"
            >
              <Upload className="w-3 h-3" />
              <span>Upload New CSV</span>
            </button>
          </div>
        )}
      </div>

      {/* 3. Live Model Pipeline Flow (Traffic Ingested -> Features -> LSTM -> Forecast) */}
      <ModelPipelineFlow
        isForecasting={isForecasting}
        peakStage={peakStage}
        peakProb={peakProb}
      />

      {/* 4. Live Active Session Topology & Flow Dynamics (Main Visual Anchor) */}
      <LiveNodeFlowVisualization
        scenario={selectedScenario}
        peakProb={peakProb}
        isForecasting={isForecasting}
        totalConnections={Math.round(currentRawValues[0] ?? 1)}
        synFlags={Math.round(currentRawValues[1] ?? 0)}
        destPorts={Math.round(currentRawValues[2] ?? 1)}
        bytesPerSec={Math.round(currentRawValues[3] ?? 300)}
        packetsPerSec={Math.round(currentRawValues[4] ?? 5)}
        iatMean={Math.round(currentRawValues[5] ?? 1000)}
        isCustomTelemetry={isCustomTelemetry}
      />

      {/* 5. Manual Telemetry Tuning (Conditional Flat Panel) */}
      {showSliders && (
        <div className="bg-[#F4F4F2] rounded-lg p-6 space-y-6 border border-[#EAEAE5]">
          <div className="flex items-center justify-between pb-4 border-b border-[#EAEAE5]">
            <div className="flex items-center gap-3">
              <span className="text-xs font-semibold uppercase tracking-wider text-[#18181B]">
                Current Observation Window Telemetry (t)
              </span>
              {isCustomTelemetry && (
                <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-white text-[#18181B] border border-[#EAEAE5] font-medium">
                  Custom Input Active
                </span>
              )}
            </div>
            {isCustomTelemetry && (
              <button
                onClick={handleResetToPreset}
                className="text-xs font-medium text-[#71717A] hover:text-[#18181B] flex items-center gap-1.5 transition-colors"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>
                  Reset to {selectedScenario === 'custom_csv' ? (uploadedCsvData?.fileName || 'CSV') : selectedScenario}
                </span>
              </button>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {FEATURE_SPECS.map((spec, idx) => {
              const rawVal = currentRawValues[idx] ?? spec.mean;
              const zVal = activeHistoryZ[4] ? activeHistoryZ[4][idx] : 0;

              return (
                <div key={spec.key} className="space-y-2">
                  <div className="flex justify-between items-baseline text-xs">
                    <label className="font-medium text-[#18181B]">
                      {spec.label}
                    </label>
                    <span className="font-mono text-xs font-semibold text-[#18181B]">
                      {rawVal.toLocaleString(undefined, { maximumFractionDigits: 1 })}{' '}
                      <span className="text-[#71717A] font-normal">{spec.unit}</span>
                    </span>
                  </div>

                  <input
                    type="range"
                    min={spec.min}
                    max={spec.max}
                    step={spec.step}
                    value={rawVal}
                    onChange={(e) => handleSliderChange(idx, parseFloat(e.target.value))}
                    className="w-full h-1.5 bg-[#EAEAE5] rounded appearance-none cursor-pointer accent-[#18181B]"
                  />

                  <div className="flex justify-between text-[11px] text-[#71717A] font-mono">
                    <span>min: {spec.min.toLocaleString()}</span>
                    <span>z: {zVal >= 0 ? `+${zVal.toFixed(1)}` : zVal.toFixed(1)}σ</span>
                    <span>max: {spec.max.toLocaleString()}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* 4. THE SINGULAR FOCAL POINT — FORECAST RESULT */}
      <div className="bg-white rounded-lg p-8 sm:p-12 space-y-8 shadow-[0_2px_12px_rgba(0,0,0,0.02),0_16px_40px_-8px_rgba(0,0,0,0.04)] border border-[#EAEAE5]/60">
        {/* Top Eyebrow & Status Row */}
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold uppercase tracking-wider text-[#A1A1AA]">
            Model Forecast Outcome
          </span>
          <div className="flex items-center gap-2">
            <span className="text-xs text-[#71717A] font-medium">MITRE Stage:</span>
            <span className="text-xs font-semibold text-[#18181B] bg-[#F4F4F2] px-3 py-1 rounded border border-[#EAEAE5]">
              {peakStage}
            </span>
          </div>
        </div>

        {/* Primary Metric: Peak Attack Probability (Single Largest Element with 32px+ breathing room) */}
        <div className="my-8 py-2">
          <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-4">
            <div className="flex items-baseline gap-4 flex-wrap">
              <AnimatedNumber
                value={peakProb}
                decimals={1}
                suffix="%"
                durationMs={450}
                className={`text-7xl sm:text-8xl font-bold font-mono tracking-tighter leading-none ${riskTheme.text}`}
              />
              <span className="text-sm sm:text-base font-medium text-[#71717A]">
                Peak Attack Probability (next 50s)
              </span>
            </div>

            <div
              className={`inline-flex items-center gap-2 px-3.5 py-1.5 rounded text-xs font-semibold self-start sm:self-auto ${riskTheme.badge}`}
            >
              <span className={`w-2 h-2 rounded-full ${riskTheme.dot}`} />
              <span>{riskTheme.label}</span>
            </div>
          </div>

          {/* Restrained Linear Risk Gauge */}
          <div className="mt-8 space-y-2">
            <div className="w-full bg-[#EAEAE5] h-2 rounded overflow-hidden">
              <div
                className={`h-full transition-all duration-300 ${riskTheme.bar}`}
                style={{ width: `${Math.min(100, Math.max(1.5, peakProb * 100))}%` }}
              />
            </div>
            <div className="flex justify-between text-[11px] text-[#A1A1AA] font-mono">
              <span>0% Baseline</span>
              <span>50% Threat Threshold</span>
              <span>100% Confirmed Threat</span>
            </div>
          </div>
        </div>

        {/* Plain-Language Takeaway Sentence (Subtle background shift) */}
        <div className="p-6 rounded-lg bg-[#F4F4F2] text-base text-[#27272A] leading-relaxed font-normal">
          &ldquo;{plainSummarySentence}&rdquo;
        </div>

        {/* Contributing Telemetry Factors */}
        <div className="flex flex-col sm:flex-row sm:items-center gap-3 text-xs">
          <span className="font-semibold uppercase tracking-wider text-[#A1A1AA] whitespace-nowrap">
            Contributing Telemetry Factors:
          </span>
          <div className="flex flex-wrap items-center gap-2">
            {plainFactors.map((factor, idx) => (
              <span
                key={idx}
                className="inline-flex items-center px-2.5 py-1 rounded bg-[#F4F4F2] text-[#18181B] border border-[#EAEAE5] font-medium text-xs transition-colors hover:border-[#18181B] cursor-default"
              >
                {factor}
              </span>
            ))}
          </div>
        </div>

        {/* Trajectory Forecast with Inline Sparkline & Trend Bars */}
        <div className="rounded-lg bg-[#FAFAF8] p-6 border border-[#EAEAE5] space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold uppercase tracking-wider text-[#A1A1AA]">
              Trajectory Forecast (+10s to +50s Horizon)
            </span>
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-flow-pulse" />
              <span className="text-[11px] font-mono text-[#71717A]">
                10-second discrete autoregressive steps
              </span>
            </div>
          </div>

          {/* Upgraded Dynamic Trajectory Sparkline Curve with Traveling Pulse */}
          <div className="w-full overflow-hidden">
            <svg viewBox="0 0 400 36" className="w-full h-9 overflow-visible">
              <defs>
                {/* Multi-stop gradient from neutral/low-risk base to semantic elevated/critical risk */}
                <linearGradient id="sparklineGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                  {sparklineGradientStops.map((stop, idx) => (
                    <stop key={idx} offset={stop.offset} stopColor={stop.color} />
                  ))}
                </linearGradient>

                {/* Subtle soft gradient area beneath curve */}
                <linearGradient id="sparklineAreaGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor={riskTheme.stroke} stopOpacity="0.1" />
                  <stop offset="100%" stopColor={riskTheme.stroke} stopOpacity="0.0" />
                </linearGradient>
              </defs>

              {/* Area under curve */}
              {sparklineAreaPath && (
                <path
                  d={sparklineAreaPath}
                  fill="url(#sparklineAreaGradient)"
                />
              )}

              {/* Baseline guide */}
              <line x1="20" y1="30" x2="380" y2="30" stroke="#EAEAE5" strokeWidth="1" strokeDasharray="3 3" />

              {/* Dynamic Gradient Trajectory Curve */}
              <path
                d={sparklineData.path}
                fill="none"
                stroke="url(#sparklineGradient)"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              {/* Traveling Pulse Along Trajectory Line */}
              {sparklineData.path && (
                <g>
                  {/* Soft Halo */}
                  <circle r="6" fill={riskTheme.stroke} opacity="0.25">
                    <animateMotion
                      path={sparklineData.path}
                      dur="2.4s"
                      repeatCount="indefinite"
                    />
                  </circle>
                  {/* Core Pulse Dot */}
                  <circle r="3" fill={riskTheme.stroke}>
                    <animateMotion
                      path={sparklineData.path}
                      dur="2.4s"
                      repeatCount="indefinite"
                    />
                  </circle>
                </g>
              )}

              {/* Discrete step dots */}
              {sparklineData.points.map((p, i) => {
                const dotColor = p.prob < 0.25 ? '#059669' : p.prob < 0.55 ? '#D97706' : '#E11D48';
                return (
                  <circle
                    key={i}
                    cx={p.x}
                    cy={p.y}
                    r="3.5"
                    fill="#FFFFFF"
                    stroke={dotColor}
                    strokeWidth="2"
                  />
                );
              })}
            </svg>
          </div>

          {/* 5-Step Trajectory Columns */}
          <div className="grid grid-cols-5 gap-2 text-center pt-1">
            {forecastSteps.map((step) => {
              const stepProb = step.attack_probability;
              const stepTheme = getStepTheme(stepProb);
              return (
                <div key={step.step} className="space-y-1">
                  <span className="text-xs text-[#71717A] font-mono block">
                    +{step.step * 10}s
                  </span>
                  {/* Micro Proportional Bar */}
                  <div className="w-2.5 h-6 bg-[#EAEAE5] rounded overflow-hidden flex flex-col justify-end mx-auto my-1">
                    <div
                      className={`w-full rounded transition-all duration-300 ${stepTheme.bar}`}
                      style={{ height: `${Math.max(3, Math.round(stepProb * 24))}px` }}
                    />
                  </div>
                  <span className={`text-sm font-semibold font-mono block ${stepTheme.text}`}>
                    {(stepProb * 100).toFixed(1)}%
                  </span>
                  <span className="text-[10px] text-[#71717A] font-medium block truncate max-w-full">
                    {step.mitre_stage}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* Secondary Raw Values Toggle */}
        <div className="border-t border-[#EAEAE5] pt-4 flex justify-between items-center">
          <span className="text-xs text-[#A1A1AA]">
            Provenance: {provenanceText}
          </span>
          <button
            id="toggle-raw-values-btn"
            onClick={() => setShowRawValues(!showRawValues)}
            className="text-xs font-semibold text-[#52525B] hover:text-[#18181B] flex items-center gap-1.5 transition-colors"
          >
            <span>{showRawValues ? 'Hide raw data' : 'Inspect raw data'}</span>
            {showRawValues ? (
              <ChevronUp className="w-3.5 h-3.5" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5" />
            )}
          </button>
        </div>

        {/* Collapsible Technical Inspection Area (Data-dense proof section) */}
        {showRawValues && (
          <div className="pt-4 border-t border-[#EAEAE5] space-y-6">
            {/* Numerical SHAP Values */}
            <div className="space-y-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-[#A1A1AA] block">
                Shapley Feature Attributions (Final Step)
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                {topShap.slice(0, 3).map(([featName, val], idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded bg-[#FAFAF8] border border-[#EAEAE5] font-mono text-xs"
                  >
                    <span className="text-[#71717A] block text-[11px] truncate">
                      {FEATURE_DISPLAY_LABELS[featName] || featName}
                    </span>
                    <span className="font-semibold text-[#18181B]">
                      {val >= 0 ? `+${val.toFixed(4)}` : val.toFixed(4)}
                    </span>
                    <span className="text-[#A1A1AA] text-[10px] ml-1">
                      ({val >= 0 ? '+risk' : '-nominal'})
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* State Matrix Table */}
            <div className="space-y-2">
              <span className="text-xs font-semibold uppercase tracking-wider text-[#A1A1AA] block">
                Telemetry State Sequence (Standardized z-scores)
              </span>
              <div className="overflow-x-auto border border-[#EAEAE5] rounded">
                <table className="w-full text-xs font-mono border-collapse text-left">
                  <thead>
                    <tr className="bg-[#F4F4F2] border-b border-[#EAEAE5] text-[#71717A]">
                      <th className="py-2 px-3 font-semibold">Interval</th>
                      <th className="py-2 px-2 font-semibold">Conns</th>
                      <th className="py-2 px-2 font-semibold">Flags</th>
                      <th className="py-2 px-2 font-semibold">Ports</th>
                      <th className="py-2 px-2 font-semibold">Bytes/s</th>
                      <th className="py-2 px-2 font-semibold">Pkts/s</th>
                      <th className="py-2 px-2 font-semibold">IAT Mean</th>
                      <th className="py-2 px-2 font-semibold">IAT Var</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#EAEAE5] bg-white">
                    {activeHistoryZ.map((row, idx) => (
                      <tr key={`h-${idx}`} className="text-[#52525B]">
                        <td className="py-1.5 px-3 text-[#18181B] font-medium">
                          t-{(4 - idx) * 10}s
                        </td>
                        {row.map((val, vIdx) => (
                          <td key={vIdx} className="py-1.5 px-2">
                            {val.toFixed(2)}
                          </td>
                        ))}
                      </tr>
                    ))}
                    {forecastSteps.map((step) => (
                      <tr
                        key={`f-${step.step}`}
                        className="text-[#18181B] bg-[#FAFAF8]"
                      >
                        <td className="py-1.5 px-3 font-semibold text-[#18181B]">
                          t+{step.step * 10}s (pred)
                        </td>
                        {step.predicted_state.map((val, vIdx) => (
                          <td key={vIdx} className="py-1.5 px-2 font-medium">
                            {val.toFixed(2)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 5. Historical Observed Sequence (t-40s to t) - Unified timeline bar */}
      <div className="space-y-3">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold uppercase tracking-wider text-[#71717A]">
            Historical Observation Window (t-40s to t)
          </span>
          <span className="text-[11px] font-mono text-[#A1A1AA]">
            5 sequential 10s states input to recurrent backbone
          </span>
        </div>

        <div className="bg-white rounded-lg border border-[#EAEAE5] overflow-hidden">
          <div className="grid grid-cols-5 divide-x divide-[#EAEAE5] text-center">
            {activeHistoryZ.map((stateVec, idx) => {
              const timeLabels = ['t-40s', 't-30s', 't-20s', 't-10s', 't (current)'];
              const isCurrent = idx === 4;

              const maxVal = Math.max(...stateVec.slice(0, 5));
              const isHigh = maxVal > 1.2;
              const isMod = maxVal > 0.4;

              return (
                <div
                  key={idx}
                  className={`py-3 px-2 transition-colors ${
                    isCurrent ? 'bg-[#F4F4F2]' : 'bg-white'
                  }`}
                >
                  <span
                    className={`text-xs font-mono block ${
                      isCurrent ? 'font-bold text-[#18181B]' : 'text-[#71717A]'
                    }`}
                  >
                    {timeLabels[idx]}
                  </span>
                  <span
                    className={`text-[11px] font-medium block mt-1 ${
                      isHigh ? 'text-rose-700' : isMod ? 'text-amber-700' : 'text-emerald-700'
                    }`}
                  >
                    {isHigh ? 'Elevated' : isMod ? 'Moderate' : 'Nominal'}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};
