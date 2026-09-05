import weightsData from './src/data/world_model_weights.json' with { type: 'json' };

// Sigmoid function
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// Tanh function
function tanh(x: number): number {
  return Math.tanh(x);
}

interface Weights {
  metadata: {
    num_features: number;
    hidden_size: number;
  };
  scaler: {
    mean: number[];
    scale: number[];
  };
  weights: {
    lstm_weight_ih: number[][]; // (256, 7)
    lstm_weight_hh: number[][]; // (256, 64)
    lstm_bias_ih: number[];     // (256)
    lstm_bias_hh: number[];     // (256)
    state_head_weight: number[][]; // (7, 64)
    state_head_bias: number[];     // (7)
    attack_head_weight: number[][]; // (1, 64)
    attack_head_bias: number[];     // (1)
  };
  mitre_thresholds: Record<string, number>;
}

const W = weightsData as Weights;
const H = W.metadata.hidden_size; // 64
const D = W.metadata.num_features; // 7

// Single forward step of LSTM
export function runLSTM(sequence: number[][]): { nextState: number[]; attackProb: number; lastHidden: number[] } {
  let h = new Array(H).fill(0);
  let c = new Array(H).fill(0);

  const w_ih = W.weights.lstm_weight_ih;
  const w_hh = W.weights.lstm_weight_hh;
  const b_ih = W.weights.lstm_bias_ih;
  const b_hh = W.weights.lstm_bias_hh;

  for (let t = 0; t < sequence.length; t++) {
    const xt = sequence[t];
    const new_h = new Array(H).fill(0);
    const new_c = new Array(H).fill(0);

    // Compute gate pre-activations: G = w_ih * xt + b_ih + w_hh * h + b_hh
    // 4 gates: i (0..63), f (64..127), g (128..191), o (192..255)
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
        const val_x = xt[j];
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

  // Head 1: state_head (Linear: 64 -> 7)
  const nextState: number[] = new Array(D).fill(0);
  const sw = W.weights.state_head_weight;
  const sb = W.weights.state_head_bias;
  for (let i = 0; i < D; i++) {
    let sum = sb[i];
    for (let j = 0; j < H; j++) {
      sum += sw[i][j] * h[j];
    }
    nextState[i] = sum;
  }

  // Head 2: attack_head (Linear: 64 -> 1, Sigmoid)
  const aw = W.weights.attack_head_weight[0];
  const ab = W.weights.attack_head_bias[0];
  let a_sum = ab;
  for (let j = 0; j < H; j++) {
    a_sum += aw[j] * h[j];
  }
  const attackProb = sigmoid(a_sum);

  return { nextState, attackProb, lastHidden: h };
}

// Test against normal scenario
import demoScenarios from './demo_scenarios.json' with { type: 'json' };

console.log("=== VERIFYING JS LSTM ENGINE AGAINST PYTHON OUTPUTS ===");
for (const scKey of ['normal', 'portscan', 'dos', 'ftp_patator'] as const) {
  console.log(`\n--- JS Scenario: ${scKey} ---`);
  const scData = (demoScenarios as any)[scKey];
  let rolling: number[][] = scData.history.map((row: number[]) => [...row]);

  for (let step = 1; step <= 5; step++) {
    const { nextState, attackProb } = runLSTM(rolling);
    console.log(`Step ${step} (+${step * 10}s): prob=${attackProb.toFixed(6)}, state=[${nextState.slice(0, 3).map(x => x.toFixed(4)).join(', ')}]...`);
    rolling = [...rolling.slice(1), nextState];
  }
}
