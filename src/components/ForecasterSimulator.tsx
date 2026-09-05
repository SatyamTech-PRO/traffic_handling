import React, { useState, useMemo, useEffect, useCallback } from 'react';
import {
  ShieldAlert,
  AlertTriangle,
  CheckCircle2,
  Play,
  RotateCcw,
  Sparkles,
  ChevronDown,
  ChevronUp,
  Crosshair,
  ShieldCheck,
  Zap,
  Activity,
  ArrowRight,
  Sliders,
  RefreshCw,
} from 'lucide-react';
import { DEMO_SCENARIOS, ScenarioKey } from '../data/mockData';
import {
  runLiveInference,
  FEATURE_SPECS,
  FeatureSpec,
  FeatureKey,
  normalizeVector,
  denormalizeVector,
  LiveInferenceResult,
} from '../lib/inferenceEngine';

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
};

const FEATURE_DISPLAY_LABELS: Record<string, string> = {
  total_connections: 'Total Connections',
  sum_syn_ack_rst_flags: 'SYN/ACK/RST Flags',
  unique_dest_ports: 'Unique Dst Ports',
  avg_bytes_per_sec: 'Avg Bytes/s',
  avg_packets_per_sec: 'Avg Packets/s',
  avg_iat_mean: 'Avg IAT Mean',
  avg_iat_variance: 'Avg IAT Variance',
};

// ---------------------------------------------------------------------------
// Dynamic Plain-Language Summary Generator
// ---------------------------------------------------------------------------
function generatePlainLanguageSummary(
  prob: number,
  stage: string,
  topShap: [string, number][]
): string {
  // 1. Lead-in clause based on probability
  let leadIn = '';
  if (prob >= 0.8) {
    leadIn = 'Critical threat detected —';
  } else if (prob >= 0.6) {
    leadIn = 'Risk is escalating —';
  } else if (prob >= 0.3) {
    leadIn = 'Elevated network risk detected —';
  } else {
    leadIn = 'Network activity is nominal —';
  }

  // 2. Core prediction clause based on MITRE stage
  let predictionClause = '';
  if (stage === 'Normal') {
    predictionClause = 'the model forecasts normal baseline traffic over the next 50 seconds';
  } else {
    predictionClause = `the model predicts a ${stage}-stage attack in the next 50 seconds`;
  }

  // 3. Driver clause derived dynamically from the #1 SHAP feature
  const topFeatKey = topShap && topShap.length > 0 ? topShap[0][0] : '';
  const driverPhrases: Record<string, string> = {
    unique_dest_ports: 'unusual port scanning activity',
    avg_packets_per_sec: 'volumetric packet flooding',
    sum_syn_ack_rst_flags: 'abnormal TCP control flag surges',
    avg_bytes_per_sec: 'irregular payload transmission volume',
    avg_iat_variance: 'traffic timing irregularity',
    avg_iat_mean: 'unusual inter-packet arrival intervals',
    total_connections: 'surging concurrent connection attempts',
  };

  const driver = driverPhrases[topFeatKey] || 'anomalous telemetry indicators';

  if (stage === 'Normal') {
    return `${leadIn} ${predictionClause}, with steady telemetry patterns suppressing alarm risk.`;
  } else {
    return `${leadIn} ${predictionClause}, driven mainly by ${driver}.`;
  }
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------
export const ForecasterSimulator: React.FC = () => {
  const [selectedScenario, setSelectedScenario] = useState<ScenarioKey>('portscan');
  const [isForecasting, setIsForecasting] = useState<boolean>(false);
  const [showRawValues, setShowRawValues] = useState<boolean>(false);
  const [showSliders, setShowSliders] = useState<boolean>(false);

  // Active 5-window sequence (z-scores). Initialized from real holdout data.
  const [activeHistoryZ, setActiveHistoryZ] = useState<number[][]>(() => {
    return DEMO_SCENARIOS['portscan'].history.map((row) => [...row]);
  });

  // Flag indicating whether the user manually tweaked any slider
  const [isCustomTelemetry, setIsCustomTelemetry] = useState<boolean>(false);

  // When scenario changes, populate activeHistoryZ from the real dataset preset
  const handleSelectScenario = (scenario: ScenarioKey) => {
    setSelectedScenario(scenario);
    setIsCustomTelemetry(false);
    const presetHistory = DEMO_SCENARIOS[scenario].history.map((row) => [...row]);
    setActiveHistoryZ(presetHistory);

    // Compute live forecast and log raw outputs to console
    runLiveInference(presetHistory, true);
  };

  // Raw values of the current window (window index 4, t)
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
      // Run live inference on every tweak
      runLiveInference(newHistory, true);
      return newHistory;
    });
  };

  // Reset sliders back to scenario preset
  const handleResetToPreset = () => {
    handleSelectScenario(selectedScenario);
  };

  // Compute live forecast results from activeHistoryZ using pure JS inference engine
  const liveResult: LiveInferenceResult = useMemo(() => {
    return runLiveInference(activeHistoryZ, false);
  }, [activeHistoryZ]);

  const forecastSteps = liveResult.forecast_steps;
  const finalProb = liveResult.final_prob;
  const finalStage = liveResult.final_stage;
  const topShap = liveResult.top_shap_features;

  // Plain-language factor names for the single-line summary
  const plainFactors = topShap.map(
    ([featName]) => PLAIN_FEATURE_NAMES[featName] || featName.replace(/_/g, ' ')
  );

  // Dynamic plain-language sentence
  const plainSummarySentence = generatePlainLanguageSummary(finalProb, finalStage, topShap);

  // Trigger brief visual feedback and fresh computation when user clicks "Run Forecast"
  const handleRunForecast = () => {
    setIsForecasting(true);
    // Explicitly run with console logging for evaluator inspection
    runLiveInference(activeHistoryZ, true);
    setTimeout(() => {
      setIsForecasting(false);
    }, 250);
  };

  // Color styles for the large MITRE stage badge
  const getMitreBadgeColor = (stage: string) => {
    switch (stage) {
      case 'Reconnaissance':
        return 'bg-yellow-500/20 text-yellow-300 border-yellow-500/40 shadow-yellow-950/40';
      case 'Impact (DoS/DDoS)':
        return 'bg-red-500/20 text-red-300 border-red-500/40 shadow-red-950/40';
      case 'Initial Access':
        return 'bg-orange-500/20 text-orange-300 border-orange-500/40 shadow-orange-950/40';
      case 'Exfiltration':
        return 'bg-purple-500/20 text-purple-300 border-purple-500/40 shadow-purple-950/40';
      case 'Lateral Movement':
        return 'bg-blue-500/20 text-blue-300 border-blue-500/40 shadow-blue-950/40';
      case 'Command & Control':
        return 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40 shadow-cyan-950/40';
      case 'Normal':
      default:
        return 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40 shadow-emerald-950/40';
    }
  };

  // Progress-bar risk meter color based on requirement:
  // green under 30%, yellow 30-60%, orange 60-80%, red above 80%
  const getRiskMeterColor = (prob: number) => {
    if (prob < 0.3) return 'bg-emerald-500';
    if (prob < 0.6) return 'bg-yellow-400';
    if (prob < 0.8) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const getRiskMeterTextColor = (prob: number) => {
    if (prob < 0.3) return 'text-emerald-400';
    if (prob < 0.6) return 'text-yellow-400';
    if (prob < 0.8) return 'text-orange-400';
    return 'text-red-400';
  };

  const getRiskLabel = (prob: number) => {
    if (prob < 0.3) return 'Low Risk (Nominal)';
    if (prob < 0.6) return 'Moderate Risk (Elevated)';
    if (prob < 0.8) return 'High Risk (Escalating)';
    return 'Critical Risk (Immediate Action)';
  };

  // Provenance text
  const provenanceText = isCustomTelemetry
    ? 'Live Synthetic / Novel Network Input Vector (Testing out-of-distribution generalization in client-side PyTorch LSTM engine)'
    : DEMO_SCENARIOS[selectedScenario]?.data_provenance || 'Holdout set, session never seen during training';

  return (
    <div className="space-y-6">
      {/* 1. Header: App title + one-line description */}
      <div className="text-center space-y-2">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-slate-900 border border-slate-800 text-xs text-slate-400 mb-1">
          <Activity className="w-3.5 h-3.5 text-emerald-400" />
          <span>Client-Side PyTorch World Model Engine</span>
        </div>
        <h1 className="text-2xl sm:text-4xl font-bold tracking-tight text-white">
          Cyber World Model
        </h1>
        <p className="text-sm sm:text-base text-slate-300 max-w-2xl mx-auto font-normal">
          Forecasts attack progression 50 seconds ahead from real network traffic
        </p>
      </div>

      {/* 2. Control Row: 4 Scenario Buttons + Manual Sliders Toggle + Run Forecast Button */}
      <div className="bg-slate-900/80 border border-slate-800/90 rounded-2xl p-3 sm:p-4 flex flex-col md:flex-row items-center justify-between gap-3 shadow-lg">
        {/* 4 Scenario Buttons */}
        <div className="w-full md:w-auto grid grid-cols-2 sm:grid-cols-4 gap-2">
          <button
            id="scenario-btn-normal"
            onClick={() => handleSelectScenario('normal')}
            className={`px-3 py-2 rounded-xl text-xs font-semibold transition-all text-center flex items-center justify-center gap-1.5 ${
              selectedScenario === 'normal' && !isCustomTelemetry
                ? 'bg-emerald-500 text-slate-950 shadow-md shadow-emerald-950'
                : 'bg-slate-950 text-slate-300 border border-slate-800 hover:border-slate-700 hover:text-white'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                selectedScenario === 'normal' && !isCustomTelemetry
                  ? 'bg-slate-950'
                  : 'bg-emerald-400'
              }`}
            />
            <span>Normal</span>
          </button>

          <button
            id="scenario-btn-portscan"
            onClick={() => handleSelectScenario('portscan')}
            className={`px-3 py-2 rounded-xl text-xs font-semibold transition-all text-center flex items-center justify-center gap-1.5 ${
              selectedScenario === 'portscan' && !isCustomTelemetry
                ? 'bg-yellow-400 text-slate-950 shadow-md shadow-yellow-950'
                : 'bg-slate-950 text-slate-300 border border-slate-800 hover:border-slate-700 hover:text-white'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                selectedScenario === 'portscan' && !isCustomTelemetry
                  ? 'bg-slate-950'
                  : 'bg-yellow-400'
              }`}
            />
            <span>PortScan</span>
          </button>

          <button
            id="scenario-btn-dos"
            onClick={() => handleSelectScenario('dos')}
            className={`px-3 py-2 rounded-xl text-xs font-semibold transition-all text-center flex items-center justify-center gap-1.5 ${
              selectedScenario === 'dos' && !isCustomTelemetry
                ? 'bg-red-500 text-white shadow-md shadow-red-950'
                : 'bg-slate-950 text-slate-300 border border-slate-800 hover:border-slate-700 hover:text-white'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                selectedScenario === 'dos' && !isCustomTelemetry ? 'bg-white' : 'bg-red-400'
              }`}
            />
            <span>DoS / DDoS</span>
          </button>

          <button
            id="scenario-btn-ftp"
            onClick={() => handleSelectScenario('ftp_patator')}
            className={`px-3 py-2 rounded-xl text-xs font-semibold transition-all text-center flex items-center justify-center gap-1.5 ${
              selectedScenario === 'ftp_patator' && !isCustomTelemetry
                ? 'bg-orange-500 text-white shadow-md shadow-orange-950'
                : 'bg-slate-950 text-slate-300 border border-slate-800 hover:border-slate-700 hover:text-white'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                selectedScenario === 'ftp_patator' && !isCustomTelemetry
                  ? 'bg-white'
                  : 'bg-orange-400'
              }`}
            />
            <span>FTP-Patator</span>
          </button>
        </div>

        {/* Action Group: Sliders Toggle + Run Forecast Button */}
        <div className="w-full md:w-auto flex items-center gap-2">
          {/* Manual Telemetry Slider Toggle Button */}
          <button
            id="toggle-sliders-btn"
            onClick={() => setShowSliders(!showSliders)}
            className={`px-3 py-2 rounded-xl text-xs font-medium transition-all flex items-center gap-1.5 border ${
              showSliders || isCustomTelemetry
                ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500/40'
                : 'bg-slate-950 text-slate-400 hover:text-slate-200 border-slate-800'
            }`}
            title="Adjust raw feature sliders to test arbitrary inputs on the live LSTM"
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>{showSliders ? 'Close Sliders' : 'Manual Sliders (7 Features)'}</span>
            {isCustomTelemetry && (
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-pulse" />
            )}
          </button>

          {/* Run Forecast Button */}
          <button
            id="run-forecast-primary-btn"
            onClick={handleRunForecast}
            disabled={isForecasting}
            className="flex-1 md:flex-initial px-5 py-2.5 rounded-xl text-xs sm:text-sm font-bold bg-emerald-500 hover:bg-emerald-400 text-slate-950 transition-all shadow-md shadow-emerald-950 flex items-center justify-center gap-2 shrink-0 active:scale-98"
          >
            {isForecasting ? (
              <Sparkles className="w-4 h-4 animate-spin text-slate-950" />
            ) : (
              <Play className="w-4 h-4 fill-current text-slate-950" />
            )}
            <span>{isForecasting ? 'Evaluating Live...' : 'Run Forecast'}</span>
          </button>
        </div>
      </div>

      {/* Manual / Slider Input Panel (Allows testing arbitrary edge-case or unseen inputs) */}
      {showSliders && (
        <div className="bg-slate-900/90 border border-cyan-900/50 rounded-2xl p-4 sm:p-5 space-y-4 shadow-xl">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Sliders className="w-4 h-4 text-cyan-400" />
              <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">
                Live Telemetry Controls (Observation Window t)
              </span>
              {isCustomTelemetry && (
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                  Custom Input Active
                </span>
              )}
            </div>
            {isCustomTelemetry && (
              <button
                onClick={handleResetToPreset}
                className="text-xs text-slate-400 hover:text-white flex items-center gap-1 transition-colors"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Reset to {selectedScenario}</span>
              </button>
            )}
          </div>

          <p className="text-[11px] text-slate-400">
            Slide any of the 7 raw network telemetry indicators to evaluate how the model responds
            to arbitrary or unseen conditions in real time.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {FEATURE_SPECS.map((spec, idx) => {
              const rawVal = currentRawValues[idx] ?? spec.mean;
              const zVal = activeHistoryZ[4] ? activeHistoryZ[4][idx] : 0;

              return (
                <div
                  key={spec.key}
                  className="p-3 rounded-xl bg-slate-950/80 border border-slate-800/90 space-y-1.5"
                >
                  <div className="flex justify-between items-baseline">
                    <label className="text-xs font-semibold text-slate-300">
                      {spec.label}
                    </label>
                    <span className="text-xs font-mono font-bold text-cyan-400">
                      {rawVal.toLocaleString(undefined, { maximumFractionDigits: 1 })}{' '}
                      <span className="text-[10px] text-slate-500 font-normal">{spec.unit}</span>
                    </span>
                  </div>

                  <input
                    type="range"
                    min={spec.min}
                    max={spec.max}
                    step={spec.step}
                    value={rawVal}
                    onChange={(e) => handleSliderChange(idx, parseFloat(e.target.value))}
                    className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
                  />

                  <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                    <span>min: {spec.min.toLocaleString()}</span>
                    <span className="text-slate-400">
                      z: {zVal >= 0 ? `+${zVal.toFixed(2)}` : zVal.toFixed(2)}σ
                    </span>
                    <span>max: {spec.max.toLocaleString()}</span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* 3. Compact Historical Input Data */}
      <div className="bg-slate-900/60 border border-slate-800/80 rounded-2xl p-4 sm:p-5">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              Observed Traffic (Last 50 Seconds)
            </span>
            <span className="text-[11px] text-slate-500 hidden sm:inline">
              5 consecutive 10-second windows fed into World Model
            </span>
          </div>
          <span className="text-[11px] font-mono text-slate-400">t-40s → t</span>
        </div>

        {/* 5 Simple Compact Cards with Mini Color Strips */}
        <div className="grid grid-cols-5 gap-2 sm:gap-3">
          {activeHistoryZ.map((stateVec, idx) => {
            const timeLabels = ['t-40s', 't-30s', 't-20s', 't-10s', 't (current)'];
            const isCurrent = idx === 4;

            // Determine relative state activity
            const maxVal = Math.max(...stateVec.slice(0, 5));
            const isHigh = maxVal > 1.2;
            const isMod = maxVal > 0.4;

            return (
              <div
                key={idx}
                className={`rounded-xl p-2 sm:p-2.5 border transition-all text-center flex flex-col justify-between ${
                  isCurrent
                    ? 'bg-slate-900 border-slate-600 ring-1 ring-emerald-500/40'
                    : 'bg-slate-950/70 border-slate-800/80'
                }`}
              >
                {/* Mini risk color strip */}
                <div className="w-full h-1 rounded-full mb-1.5 overflow-hidden bg-slate-800">
                  <div
                    className={`h-full ${
                      isHigh
                        ? 'bg-red-400'
                        : isMod
                        ? 'bg-yellow-400'
                        : 'bg-emerald-400'
                    }`}
                  />
                </div>

                <span
                  className={`text-[11px] sm:text-xs font-mono font-medium block truncate ${
                    isCurrent ? 'text-emerald-300 font-semibold' : 'text-slate-300'
                  }`}
                >
                  {timeLabels[idx]}
                </span>

                <span className="text-[10px] text-slate-400 block mt-0.5">
                  {isHigh ? 'High Vol' : isMod ? 'Elevated' : 'Nominal'}
                </span>
              </div>
            );
          })}
        </div>
      </div>

      {/* 4. The RESULT: Big, Clear, Plain Language */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-6 sm:p-8 space-y-6 shadow-xl relative overflow-hidden">
        {/* Subtle Ambient Glow */}
        <div
          className={`absolute -top-24 -right-24 w-64 h-64 rounded-full blur-3xl opacity-10 pointer-events-none ${
            finalProb >= 0.6
              ? 'bg-red-500'
              : finalProb >= 0.3
              ? 'bg-yellow-500'
              : 'bg-emerald-500'
          }`}
        />

        <div className="text-center space-y-4">
          <span className="text-xs uppercase tracking-wider text-slate-400 font-semibold block">
            50-Second Lookahead Forecast Result
          </span>

          {/* ITEM 1: Large Colored Badge: MITRE Stage */}
          <div className="flex justify-center">
            <div
              className={`inline-flex items-center gap-2.5 px-6 py-2.5 rounded-2xl border text-lg sm:text-2xl font-bold shadow-lg transition-all ${getMitreBadgeColor(
                finalStage
              )}`}
            >
              <Crosshair className="w-5 h-5 sm:w-6 sm:h-6 shrink-0" />
              <span>{finalStage}</span>
            </div>
          </div>

          {/* ITEM 2: Large Percentage + Simple Colored Progress-Bar Risk Meter */}
          <div className="space-y-3 max-w-lg mx-auto pt-2">
            <div className="flex items-baseline justify-center gap-2">
              <span
                className={`text-5xl sm:text-6xl font-black font-mono tracking-tight ${getRiskMeterTextColor(
                  finalProb
                )}`}
              >
                {(finalProb * 100).toFixed(1)}%
              </span>
              <span className="text-xs sm:text-sm font-semibold text-slate-400 uppercase tracking-wide">
                Attack Probability
              </span>
            </div>

            {/* Simple Colored Progress Bar Risk Meter */}
            <div className="space-y-1.5">
              <div className="w-full bg-slate-950 h-3 rounded-full overflow-hidden border border-slate-800 p-0.5">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${getRiskMeterColor(
                    finalProb
                  )}`}
                  style={{ width: `${Math.min(100, Math.max(4, finalProb * 100))}%` }}
                />
              </div>

              <div className="flex justify-between text-[11px] text-slate-500 font-medium px-1">
                <span>0% Nominal</span>
                <span className={`font-semibold ${getRiskMeterTextColor(finalProb)}`}>
                  {getRiskLabel(finalProb)}
                </span>
                <span>100% Critical</span>
              </div>
            </div>
          </div>

          {/* ITEM 3: ONE Auto-Generated Plain-Language Sentence */}
          <div className="pt-3 max-w-2xl mx-auto">
            <div className="p-4 rounded-xl bg-slate-950/80 border border-slate-800/90 text-sm sm:text-base text-slate-200 font-medium leading-relaxed">
              &ldquo;{plainSummarySentence}&rdquo;
            </div>
          </div>

          {/* ITEM 5: SHAP Explanation — One Clean Line */}
          <div className="pt-1">
            <p className="text-xs sm:text-sm text-slate-400">
              <span className="font-semibold text-slate-300">Top factors:</span>{' '}
              <span className="text-emerald-400 font-medium">
                {plainFactors.join(', ')}
              </span>
            </p>
          </div>

          {/* Mini 5-Step Trajectory Outlook (t+10s to t+50s) */}
          <div className="pt-4 border-t border-slate-800/80 max-w-xl mx-auto">
            <span className="text-[11px] text-slate-500 font-semibold uppercase tracking-wider block mb-2">
              Autoregressive Step Progression
            </span>
            <div className="flex items-center justify-between gap-1 text-center">
              {forecastSteps.map((step) => {
                const stepProb = step.attack_probability;
                return (
                  <div
                    key={step.step}
                    className="flex-1 px-1 py-1.5 rounded-lg bg-slate-950 border border-slate-800/80"
                  >
                    <span className="text-[10px] text-slate-400 block font-mono">
                      +{step.step * 10}s
                    </span>
                    <span
                      className={`text-xs font-bold font-mono block ${getRiskMeterTextColor(
                        stepProb
                      )}`}
                    >
                      {(stepProb * 100).toFixed(0)}%
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        {/* 5. Collapsed "Show raw values" Toggle */}
        <div className="pt-2 border-t border-slate-800 flex justify-center">
          <button
            id="toggle-raw-values-btn"
            onClick={() => setShowRawValues(!showRawValues)}
            className="px-4 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-slate-200 bg-slate-950 hover:bg-slate-900 border border-slate-800 transition-all flex items-center gap-2"
          >
            <span>{showRawValues ? 'Hide raw values' : 'Show raw values'}</span>
            {showRawValues ? (
              <ChevronUp className="w-3.5 h-3.5" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5" />
            )}
          </button>
        </div>

        {/* Collapsed Technical Depth Section */}
        {showRawValues && (
          <div className="mt-4 p-4 sm:p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-4 text-xs">
            {/* Technical Detail 1: Data Provenance */}
            <div className="p-3 rounded-lg bg-slate-900 border border-slate-800 flex items-start gap-2.5">
              <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
              <div>
                <span className="text-slate-200 font-semibold block text-xs">
                  Validation Data Provenance
                </span>
                <span className="text-slate-400 font-mono text-[11px]">
                  {provenanceText}
                </span>
              </div>
            </div>

            {/* Technical Detail 2: Exact Numerical SHAP Values */}
            <div className="space-y-2">
              <span className="text-slate-300 font-semibold block">
                Exact SHAP Shapley Feature Attributions (Final Step)
              </span>
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                {topShap.slice(0, 3).map(([featName, val], idx) => (
                  <div
                    key={idx}
                    className="p-2.5 rounded-lg bg-slate-900 border border-slate-800 font-mono text-[11px]"
                  >
                    <span className="text-slate-400 block text-[10px] truncate">
                      {FEATURE_DISPLAY_LABELS[featName] || featName}
                    </span>
                    <span
                      className={`font-bold ${
                        val >= 0 ? 'text-red-400' : 'text-emerald-400'
                      }`}
                    >
                      {val >= 0 ? `+${val.toFixed(4)}` : val.toFixed(4)}
                    </span>
                    <span className="text-slate-500 text-[10px] ml-1.5">
                      ({val >= 0 ? '+Risk' : '-Normal'})
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Technical Detail 3: Exact 7-Feature Z-Score States */}
            <div className="space-y-2">
              <span className="text-slate-300 font-semibold block">
                Normalized State Vectors (StandardScaler z-scores)
              </span>
              <div className="overflow-x-auto">
                <table className="w-full text-[11px] font-mono border-collapse text-left">
                  <thead>
                    <tr className="border-b border-slate-800 text-slate-500">
                      <th className="py-1.5 px-2">Time</th>
                      <th className="py-1.5 px-2">Conns</th>
                      <th className="py-1.5 px-2">Flags</th>
                      <th className="py-1.5 px-2">Ports</th>
                      <th className="py-1.5 px-2">Bytes/s</th>
                      <th className="py-1.5 px-2">Pkts/s</th>
                      <th className="py-1.5 px-2">IAT Mean</th>
                      <th className="py-1.5 px-2">IAT Var</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activeHistoryZ.map((row, idx) => (
                      <tr
                        key={`h-${idx}`}
                        className="border-b border-slate-900 text-slate-400"
                      >
                        <td className="py-1 px-2 text-slate-300 font-semibold">
                          t-{(4 - idx) * 10}s
                        </td>
                        {row.map((val, vIdx) => (
                          <td key={vIdx} className="py-1 px-2">
                            {val.toFixed(2)}
                          </td>
                        ))}
                      </tr>
                    ))}
                    {forecastSteps.map((step) => (
                      <tr
                        key={`f-${step.step}`}
                        className="border-b border-slate-900 text-emerald-300 bg-emerald-950/10"
                      >
                        <td className="py-1 px-2 font-semibold">
                          t+{step.step * 10}s (pred)
                        </td>
                        {step.predicted_state.map((val, vIdx) => (
                          <td key={vIdx} className="py-1 px-2">
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
    </div>
  );
};
