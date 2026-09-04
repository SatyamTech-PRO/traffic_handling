import React from 'react';
import { Award, ShieldAlert, Cpu, Sparkles, Brain, CheckCircle2, ChevronRight } from 'lucide-react';

export const JudgeGuide: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-900 to-emerald-950/30 border border-slate-800 rounded-xl p-6">
        <div className="flex items-center gap-2 mb-2">
          <Award className="w-5 h-5 text-emerald-400" />
          <h2 className="text-base font-bold text-slate-100">
            Hackathon Judge Defense & World Model Architecture
          </h2>
        </div>
        <p className="text-xs text-slate-400 max-w-3xl leading-relaxed">
          Use these architectural principles and quantitative defenses during your pitch to demonstrate deep mastery of reinforcement learning, temporal MDP state modeling, and real-world network data hygiene.
        </p>
      </div>

      {/* Core Defense Pillars */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {/* Pillar 1 */}
        <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
          <div className="flex items-center gap-2 text-emerald-400 font-semibold text-sm">
            <Cpu className="w-4 h-4" />
            <span>1. Why Discretize into Fixed 10s Time Windows?</span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            <strong className="text-slate-100">The Problem:</strong> Raw network packets and NetFlow records arrive asynchronously with Poisson or Pareto inter-arrival times. Machine learning world models (like Dreamer, RSSMs, or Transformers) require synchronized, uniform time steps <span className="font-mono text-emerald-400">t, t+1, t+2</span>.
          </p>
          <p className="text-xs text-slate-400 leading-relaxed">
            <strong className="text-slate-100">The Defense:</strong> A 10-second window captures macro-level thermodynamic equilibrium of network sessions. It is long enough to observe full TCP handshakes and teardowns, but short enough to immediately detect fast DDoS spikes and rapid horizontal port scans.
          </p>
        </div>

        {/* Pillar 2 */}
        <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
          <div className="flex items-center gap-2 text-emerald-400 font-semibold text-sm">
            <ShieldAlert className="w-4 h-4" />
            <span>2. Why StandardScaler Instead of MinMaxScaler?</span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            <strong className="text-slate-100">The Problem:</strong> Network traffic distributions are heavily heavy-tailed and bursty (e.g., during a SYN flood or volumetric DDoS, packets/sec can jump 1,000×).
          </p>
          <p className="text-xs text-slate-400 leading-relaxed">
            <strong className="text-slate-100">The Defense:</strong> If you use <code className="text-slate-300">MinMaxScaler[0, 1]</code>, a single volumetric attack peak crushes 99% of normal benign traffic into a microscopic range <span className="font-mono text-amber-300">[0.000, 0.005]</span>, destroying gradient signal. <code className="text-emerald-300">StandardScaler</code> normalizes by variance, preserving subtle benign fluctuations while representing anomalies as multi-sigma outliers (<span className="font-mono text-emerald-400">z &gt; 3.0</span>).
          </p>
        </div>

        {/* Pillar 3 */}
        <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
          <div className="flex items-center gap-2 text-emerald-400 font-semibold text-sm">
            <Brain className="w-4 h-4" />
            <span>3. Strict Separation of Target Labels</span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            <strong className="text-slate-100">The Problem:</strong> In naive pipelines, practitioners accidentally include the ground truth attack label inside feature normalization, causing catastrophic data leakage.
          </p>
          <p className="text-xs text-slate-400 leading-relaxed">
            <strong className="text-slate-100">The Defense:</strong> The 7-dimensional continuous state vector <span className="font-mono text-emerald-400">s_t</span> contains strictly observable telemetry (flow volume, flags, port count, throughput, jitter). The <span className="font-mono text-purple-400">attack_fraction</span> is held out purely as an evaluation reward or supervisory target <span className="font-mono text-purple-400">y_t</span>.
          </p>
        </div>

        {/* Pillar 4 */}
        <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
          <div className="flex items-center gap-2 text-emerald-400 font-semibold text-sm">
            <Sparkles className="w-4 h-4" />
            <span>4. Production Inference Readiness (joblib)</span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            <strong className="text-slate-100">The Problem:</strong> Hackathon models often fail when deploying in live environments because normalization parameters are lost in notebook memory.
          </p>
          <p className="text-xs text-slate-400 leading-relaxed">
            <strong className="text-slate-100">The Defense:</strong> The fitted <code className="text-emerald-300">StandardScaler</code> instance is permanently saved to disk with <code className="text-emerald-300">joblib.dump('scaler.joblib')</code>. In live production or streaming deployment, any new 10s packet window is transformed using the exact precomputed mean and variance.
          </p>
        </div>

        {/* Pillar 5 */}
        <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
          <div className="flex items-center gap-2 text-emerald-400 font-semibold text-sm">
            <ShieldAlert className="w-4 h-4" />
            <span>5. MITRE ATT&CK Grounding & Chronological Session-Split Audit</span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            <strong className="text-slate-100">The Problem:</strong> Naive random shuffling of 10-second sliding windows causes data leakage, as adjacent windows from the same burst appear on both sides of the split. Furthermore, volumetric attacks are frequently mislabeled as initial access.
          </p>
          <p className="text-xs text-slate-400 leading-relaxed">
            <strong className="text-slate-100">The Defense:</strong> We implemented a <strong className="text-slate-200">chronological session-level split</strong> in <code className="text-emerald-300">auto_calibrate_thresholds.py</code> and an 80/20 chronological timeline split for the LSTM. Multi-session BENIGN traffic proves 100% specificity across truly unseen time blocks, while single-burst attacks are audited with explicit caveat warnings for demo integrity.
          </p>
          <div className="p-2.5 rounded-lg bg-emerald-950/40 border border-emerald-800/60 text-emerald-300 text-xs font-medium">
            Scenarios are real inference outputs from the trained model on held-out network sessions never used for calibration.
          </div>
          <p className="text-xs text-slate-400 leading-relaxed">
            <strong className="text-cyan-400 font-medium">Deliberate Tactic Extension:</strong> In standard MITRE ATT&CK, volumetric flooding attacks (DDoS, DoS Hulk) belong to the <span className="text-emerald-300 font-medium">Impact tactic (TA0040, T1498/T1499)</span>, not Initial Access. We deliberately extended the 5 SIH stages with &ldquo;Impact (DoS/DDoS)&rdquo; to ensure defensible, domain-accurate threat classification.
          </p>
        </div>

        {/* Pillar 6 */}
        <div className="p-5 rounded-xl bg-slate-950 border border-slate-800 space-y-3">
          <div className="flex items-center gap-2 text-emerald-400 font-semibold text-sm">
            <Brain className="w-4 h-4" />
            <span>6. Explainable AI: SHAP Attribution for SOC Trust</span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            <strong className="text-slate-100">The Problem:</strong> Complex deep recurrent networks (like LSTMs) operate as black boxes, causing alert fatigue and distrust from incident responders.
          </p>
          <p className="text-xs text-slate-400 leading-relaxed">
            <strong className="text-slate-100">The Defense:</strong> Using <code className="text-emerald-300">shap.KernelExplainer</code>, our framework derives exact local Shapley attributions for each alarm, immediately proving to the operator which top 3 features (e.g. flag surges vs port proliferation) caused the warning.
          </p>
        </div>
      </div>

      {/* World Model Next Steps */}
      <div className="p-5 rounded-xl bg-slate-950 border border-slate-800">
        <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-300 mb-3 flex items-center gap-2">
          <CheckCircle2 className="w-4 h-4 text-emerald-400" />
          <span>Next Steps: Training Your World Model with states.npy</span>
        </h3>

        <div className="space-y-2 text-xs text-slate-300">
          <div className="flex items-start gap-2">
            <ChevronRight className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
            <span>
              <strong>State Transition Dynamics:</strong> Train an auto-regressive model or Recurrent State-Space Model (RSSM) to predict the next environment state: <span className="font-mono text-emerald-400">P(s_&#123;t+1&#125; | s_t, a_t)</span> where <span className="font-mono text-slate-400">a_t</span> is a cyber defense action (e.g., rate-limit, firewall block, honeypot redirect).
            </span>
          </div>
          <div className="flex items-start gap-2">
            <ChevronRight className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
            <span>
              <strong>Simulated Cyber Range:</strong> Once the transition model is trained, an autonomous Blue Team agent can practice thousands of defensive episodes in a purely hallucinated environment without touching production servers!
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};
