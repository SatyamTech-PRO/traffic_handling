import weightsData from '../data/world_model_weights.json';

export interface WeightsSchema {
  metadata: {
    model_name: string;
    framework: string;
    num_features: number;
    hidden_size: number;
    num_layers: number;
    gate_order: string[];
    feature_names: string[];
  };
  scaler: {
    mean: number[];
    scale: number[];
    var: number[];
  };
  weights: {
    lstm_weight_ih: number[][]; // (256, 14)
    lstm_weight_hh: number[][]; // (256, 64)
    lstm_bias_ih: number[];     // (256)
    lstm_bias_hh: number[];     // (256)
    state_head_weight: number[][]; // (14, 64)
    state_head_bias: number[];     // (14)
    attack_head_weight: number[][]; // (1, 64)
    attack_head_bias: number[];     // (1)
  };
  mitre_thresholds: Record<string, number>;
}

export const MODEL_WEIGHTS = weightsData as WeightsSchema;
const H = MODEL_WEIGHTS.metadata.hidden_size; // 64
const D = MODEL_WEIGHTS.metadata.num_features; // 14

export const FEATURE_KEYS = [
  'total_connections',
  'sum_syn_ack_rst_flags',
  'unique_dest_ports',
  'avg_bytes_per_sec',
  'avg_packets_per_sec',
  'avg_iat_mean',
  'avg_iat_variance',
] as const;

export type FeatureKey = (typeof FEATURE_KEYS)[number];

export interface FeatureSpec {
  key: FeatureKey;
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  mean: number;
  scale: number;
}

export const FEATURE_SPECS: FeatureSpec[] = [
  {
    key: 'total_connections',
    label: 'Total Connections',
    unit: 'conns / 10s',
    min: 5,
    max: 50,
    step: 1,
    mean: MODEL_WEIGHTS.scaler.mean[0],
    scale: MODEL_WEIGHTS.scaler.scale[0],
  },
  {
    key: 'sum_syn_ack_rst_flags',
    label: 'SYN/ACK/RST Flags',
    unit: 'flags / 10s',
    min: 0,
    max: 3500,
    step: 10,
    mean: MODEL_WEIGHTS.scaler.mean[1],
    scale: MODEL_WEIGHTS.scaler.scale[1],
  },
  {
    key: 'unique_dest_ports',
    label: 'Unique Dst Ports',
    unit: 'ports / 10s',
    min: 1,
    max: 50,
    step: 1,
    mean: MODEL_WEIGHTS.scaler.mean[2],
    scale: MODEL_WEIGHTS.scaler.scale[2],
  },
  {
    key: 'avg_bytes_per_sec',
    label: 'Avg Bytes/s',
    unit: 'B/s',
    min: 0,
    max: 300000,
    step: 500,
    mean: MODEL_WEIGHTS.scaler.mean[3],
    scale: MODEL_WEIGHTS.scaler.scale[3],
  },
  {
    key: 'avg_packets_per_sec',
    label: 'Avg Packets/s',
    unit: 'pkts/s',
    min: 0,
    max: 2500,
    step: 5,
    mean: MODEL_WEIGHTS.scaler.mean[4],
    scale: MODEL_WEIGHTS.scaler.scale[4],
  },
  {
    key: 'avg_iat_mean',
    label: 'Avg IAT Mean',
    unit: 'μs',
    min: 100,
    max: 25000,
    step: 100,
    mean: MODEL_WEIGHTS.scaler.mean[5],
    scale: MODEL_WEIGHTS.scaler.scale[5],
  },
  {
    key: 'avg_iat_variance',
    label: 'Avg IAT Variance',
    unit: 'μs²',
    min: 10000,
    max: 20000000,
    step: 10000,
    mean: MODEL_WEIGHTS.scaler.mean[6],
    scale: MODEL_WEIGHTS.scaler.scale[6],
  },
];

// ---------------------------------------------------------------------------
// Math Primitives
// ---------------------------------------------------------------------------
export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export function tanh(x: number): number {
  return Math.tanh(x);
}

// ---------------------------------------------------------------------------
// Normalization Utilities (StandardScaler)
// ---------------------------------------------------------------------------
export function normalizeVector(rawVector: number[]): number[] {
  const { mean, scale } = MODEL_WEIGHTS.scaler;
  return rawVector.map((val, idx) => {
    const s = scale[idx] ?? 1;
    const m = mean[idx] ?? 0;
    return (val - m) / s;
  });
}

export function denormalizeVector(zVector: number[]): number[] {
  const { mean, scale } = MODEL_WEIGHTS.scaler;
  return zVector.map((z, idx) => {
    const s = scale[idx] ?? 1;
    const m = mean[idx] ?? 0;
    return z * s + m;
  });
}

// ---------------------------------------------------------------------------
// MITRE ATT&CK Tactic Mapping (Calibrated Empirical Logic from Python model)
// ---------------------------------------------------------------------------
export function mapToMitreStage(stateVector: number[]): string {
  const t = MODEL_WEIGHTS.mitre_thresholds;

  const total_conns = stateVector[0];
  const flags = stateVector[1];
  const unique_ports = stateVector[2];
  const bytes_sec = stateVector[3];
  const pkts_sec = stateVector[4];
  const iat_var = stateVector[6];

  // 1. Impact (DoS/DDoS) - volumetric flag or packet flood
  if (flags >= t.THRESH_IMPACT_FLAGS || pkts_sec >= t.THRESH_IMPACT_PKTS) {
    return 'Impact (DoS/DDoS)';
  }

  // 2. Reconnaissance (PortScan) - destination port sweep
  if (unique_ports >= t.THRESH_RECON_PORTS) {
    return 'Reconnaissance';
  }

  // 3. Initial Access (FTP-Patator)
  if (bytes_sec <= t.THRESH_BRUTE_BYTES_MAX && unique_ports <= t.THRESH_BRUTE_PORTS_MAX) {
    return 'Initial Access';
  }

  // 4. Exfiltration
  if (bytes_sec >= t.THRESH_EXFIL_BYTES && unique_ports <= t.THRESH_EXFIL_PORTS_MAX) {
    return 'Exfiltration';
  }

  // 5. Lateral Movement
  if (total_conns >= t.THRESH_LATERAL_CONNS && unique_ports >= t.THRESH_LATERAL_PORTS) {
    return 'Lateral Movement';
  }

  // 6. Command & Control
  if (iat_var <= t.THRESH_C2_IAT_VAR_MAX && bytes_sec <= t.THRESH_C2_BYTES_MAX && pkts_sec > -0.5) {
    return 'Command & Control';
  }

  return 'Normal';
}

// ---------------------------------------------------------------------------
// Pure JS PyTorch LSTM Forward Pass
// ---------------------------------------------------------------------------
export interface LSTMForwardResult {
  nextState: number[];
  attackProb: number;
  lastHidden: number[];
}

export function runLSTMForward(sequenceZ: number[][]): LSTMForwardResult {
  let h = new Array(H).fill(0);
  let c = new Array(H).fill(0);

  const w_ih = MODEL_WEIGHTS.weights.lstm_weight_ih;
  const w_hh = MODEL_WEIGHTS.weights.lstm_weight_hh;
  const b_ih = MODEL_WEIGHTS.weights.lstm_bias_ih;
  const b_hh = MODEL_WEIGHTS.weights.lstm_bias_hh;

  for (let t = 0; t < sequenceZ.length; t++) {
    const xt = sequenceZ[t];
    const new_h = new Array(H).fill(0);
    const new_c = new Array(H).fill(0);

    for (let k = 0; k < H; k++) {
      const idx_i = k;
      const idx_f = k + H;
      const idx_g = k + 2 * H;
      const idx_o = k + 3 * H;

      let sum_i = b_ih[idx_i] + b_hh[idx_i];
      let sum_f = b_ih[idx_f] + b_hh[idx_f];
      let sum_g = b_ih[idx_g] + b_hh[idx_g];
      let sum_o = b_ih[idx_o] + b_hh[idx_o];

      for (let j = 0; j < D; j++) {
        const val_x = xt && xt[j] !== undefined ? xt[j] : 0;
        sum_i += w_ih[idx_i][j] * val_x;
        sum_f += w_ih[idx_f][j] * val_x;
        sum_g += w_ih[idx_g][j] * val_x;
        sum_o += w_ih[idx_o][j] * val_x;
      }

      for (let j = 0; j < H; j++) {
        const val_h = h[j];
        sum_i += w_hh[idx_i][j] * val_h;
        sum_f += w_hh[idx_f][j] * val_h;
        sum_g += w_hh[idx_g][j] * val_h;
        sum_o += w_hh[idx_o][j] * val_h;
      }

      const gate_i = sigmoid(sum_i);
      const gate_f = sigmoid(sum_f);
      const gate_g = tanh(sum_g);
      const gate_o = sigmoid(sum_o);

      const cell_k = gate_f * c[k] + gate_i * gate_g;
      const hidden_k = gate_o * tanh(cell_k);

      new_c[k] = cell_k;
      new_h[k] = hidden_k;
    }

    h = new_h;
    c = new_c;
  }

  // Head 1: state_head (Linear: 64 -> 14)
  const nextState: number[] = new Array(D).fill(0);
  const sw = MODEL_WEIGHTS.weights.state_head_weight;
  const sb = MODEL_WEIGHTS.weights.state_head_bias;
  for (let i = 0; i < D; i++) {
    let sum = sb[i];
    for (let j = 0; j < H; j++) {
      sum += sw[i][j] * h[j];
    }
    nextState[i] = sum;
  }

  // Head 2: attack_head (Linear: 64 -> 1, Sigmoid)
  const aw = MODEL_WEIGHTS.weights.attack_head_weight[0];
  const ab = MODEL_WEIGHTS.weights.attack_head_bias[0];
  let a_sum = ab;
  for (let j = 0; j < H; j++) {
    a_sum += aw[j] * h[j];
  }
  const attackProb = sigmoid(a_sum);

  return { nextState, attackProb, lastHidden: h };
}

// ---------------------------------------------------------------------------
// Perturbation Sensitivity Feature Attribution (Dynamic SHAP-Equivalent)
// ---------------------------------------------------------------------------
export function computeFeatureAttribution(sequenceZ: number[][]): [string, number][] {
  const baseResult = runLSTMForward(sequenceZ);
  const baseProb = baseResult.attackProb;

  const attributions: [string, number][] = [];
  const featNames = MODEL_WEIGHTS.metadata.feature_names || FEATURE_KEYS;

  for (let featIdx = 0; featIdx < D; featIdx++) {
    // Perturb sequence by resetting feature to normalized zero (mean)
    const perturbedSeq = sequenceZ.map((row) => {
      const copy = [...row];
      copy[featIdx] = 0.0;
      return copy;
    });

    const perturbedResult = runLSTMForward(perturbedSeq);
    const impact = baseProb - perturbedResult.attackProb;
    const featKey = featNames[featIdx] || (FEATURE_KEYS as readonly string[])[featIdx] || `feature_${featIdx}`;
    attributions.push([featKey, impact]);
  }

  // Sort descending by absolute attribution magnitude
  attributions.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  return attributions;
}

// ---------------------------------------------------------------------------
// Autoregressive Closed-Loop Rollout (+10s to +50s)
// ---------------------------------------------------------------------------
export interface LiveForecastStep {
  step: number;
  predicted_state: number[];
  attack_probability: number;
  mitre_stage: string;
}

export interface LiveInferenceResult {
  forecast_steps: LiveForecastStep[];
  final_prob: number;
  final_stage: string;
  top_shap_features: [string, number][];
}

export function runLiveInference(
  initial5WindowsZ: number[][],
  logToConsole = true
): LiveInferenceResult {
  let rolling = initial5WindowsZ.map((row) => [...row]);
  const steps: LiveForecastStep[] = [];

  if (logToConsole) {
    console.log('═══════════════════════════════════════════════════════════════');
    console.log(' [Live JS Inference Engine] Closed-Loop Autoregressive Rollout');
    console.log('═══════════════════════════════════════════════════════════════');
  }

  for (let step = 1; step <= 5; step++) {
    const { nextState, attackProb } = runLSTMForward(rolling);
    const stage = mapToMitreStage(nextState);

    steps.push({
      step,
      predicted_state: nextState,
      attack_probability: attackProb,
      mitre_stage: stage,
    });

    if (logToConsole) {
      console.log(
        ` [Live Step +${step * 10}s] P(Attack)=${(attackProb * 100).toFixed(2)}% | MITRE=${stage.padEnd(18)} | Next State (z)=`,
        nextState.map((x) => Number(x.toFixed(3)))
      );
    }

    // Autoregressive update: shift window and push predicted state
    rolling = [...rolling.slice(1), nextState];
  }

  const finalStep = steps[steps.length - 1];
  const finalProb = finalStep.attack_probability;
  const finalStage = finalStep.mitre_stage;

  // Compute live feature attribution on the input sequence
  const topShap = computeFeatureAttribution(initial5WindowsZ);

  if (logToConsole) {
    console.log(' [Live Explanations] Top 3 Drivers:', topShap.slice(0, 3));
    console.log('═══════════════════════════════════════════════════════════════');
  }

  return {
    forecast_steps: steps,
    final_prob: finalProb,
    final_stage: finalStage,
    top_shap_features: topShap,
  };
}
