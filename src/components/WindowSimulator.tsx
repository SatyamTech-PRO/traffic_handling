import React, { useState, useMemo } from 'react';
import { NetworkFlow, StateVectorWindow } from '../types';
import { generateSyntheticFlows } from '../data/mockData';
import { aggregateFlowsIntoWindows, generateCsvString } from '../utils/aggregator';
import {
  Activity,
  Download,
  RotateCcw,
  PlusCircle,
  AlertTriangle,
  Server,
  Layers,
  ArrowRight,
  TrendingUp,
  Cpu
} from 'lucide-react';

export const WindowSimulator: React.FC = () => {
  const [flows, setFlows] = useState<NetworkFlow[]>(() => generateSyntheticFlows(70));
  const [selectedWindowIdx, setSelectedWindowIdx] = useState<number | null>(1);
  const [windowSec, setWindowSec] = useState<number>(10);

  // Compute aggregated state vector windows
  const windows = useMemo(() => {
    return aggregateFlowsIntoWindows(flows, windowSec);
  }, [flows, windowSec]);

  // Selected window details
  const selectedWindow = useMemo(() => {
    if (!selectedWindowIdx) return windows[0] || null;
    return windows.find((w) => w.windowIndex === selectedWindowIdx) || windows[0] || null;
  }, [windows, selectedWindowIdx]);

  // Flows belonging to selected window
  const selectedWindowFlows = useMemo(() => {
    if (!selectedWindow) return [];
    return flows.filter((f) => {
      const timeStr = new Date(f.timestamp).toTimeString().substring(0, 8);
      return timeStr >= selectedWindow.windowStartTime && timeStr < selectedWindow.windowEndTime;
    });
  }, [flows, selectedWindow]);

  // Handler to inject attack
  const injectAttackBurst = (attackType: 'DDoS' | 'PortScan') => {
    const lastTimestamp = flows.length > 0 ? new Date(flows[flows.length - 1].timestamp) : new Date();
    const newFlows: NetworkFlow[] = [];

    for (let i = 0; i < 15; i++) {
      const t = new Date(lastTimestamp.getTime() + (i * 0.4) * 1000);
      const isPortScan = attackType === 'PortScan';

      newFlows.push({
        id: `inject-${Date.now()}-${i}`,
        timestamp: t.toISOString().replace('T', ' ').substring(0, 19),
        destPort: isPortScan ? 1024 + Math.floor(Math.random() * 50000) : 80,
        flowDuration: isPortScan ? 800 : 900000,
        totalFwdPackets: isPortScan ? 1 : 120,
        totalBackwardPackets: isPortScan ? 0 : 60,
        flowBytesPerSec: isPortScan ? 400 : 250000,
        flowPacketsPerSec: isPortScan ? 30 : 1500,
        synFlagCount: isPortScan ? 1 : 15,
        ackFlagCount: isPortScan ? 0 : 80,
        rstFlagCount: isPortScan ? 1 : 2,
        fwdIatMean: isPortScan ? 80 : 30,
        fwdIatStd: isPortScan ? 15 : 10,
        label: attackType,
      });
    }

    setFlows((prev) => [...prev, ...newFlows].sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()));
  };

  const handleReset = () => {
    setFlows(generateSyntheticFlows(70));
    setSelectedWindowIdx(1);
  };

  const downloadCsv = () => {
    const csvContent = generateCsvString(windows);
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'states.csv';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const downloadNpy = () => {
    // Generate mock .npy file download for demo
    // The Python script saves authentic numpy array via np.save("states.npy", X_scaled)
    const rawMatrix = windows.map((w) => [
      w.normTotalConnections ?? 0,
      w.normSumFlags ?? 0,
      w.normUniquePorts ?? 0,
      w.normAvgBytes ?? 0,
      w.normAvgPackets ?? 0,
      w.normAvgIatMean ?? 0,
      w.normAvgIatVariance ?? 0,
    ]);

    const jsonStr = JSON.stringify(rawMatrix, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'states_matrix_preview.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Simulation Controls Bar */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-sm font-semibold text-slate-100 flex items-center gap-2">
            <Cpu className="w-4 h-4 text-emerald-400" />
            <span>10-Second State Vector Simulator</span>
          </h2>
          <p className="text-xs text-slate-400 mt-0.5">
            Observing {flows.length} asynchronous network flows discretized into {windows.length} sequential World Model states.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            id="inject-ddos-btn"
            onClick={() => injectAttackBurst('DDoS')}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-red-500/10 text-red-400 border border-red-500/30 hover:bg-red-500/20 transition"
          >
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>+ Inject DDoS Burst</span>
          </button>

          <button
            id="inject-portscan-btn"
            onClick={() => injectAttackBurst('PortScan')}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/30 hover:bg-amber-500/20 transition"
          >
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>+ Inject PortScan</span>
          </button>

          <button
            id="reset-simulation-btn"
            onClick={handleReset}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 text-slate-300 hover:bg-slate-700 transition"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset</span>
          </button>

          <div className="h-4 w-px bg-slate-800 mx-1"></div>

          <button
            id="download-sim-csv-btn"
            onClick={downloadCsv}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-emerald-500 text-slate-950 hover:bg-emerald-400 transition shadow-sm"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export states.csv</span>
          </button>
        </div>
      </div>

      {/* Main Grid: State Vectors List & Selected State Details */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Timeline of 10s Windows */}
        <div className="lg:col-span-5 space-y-4">
          <div className="bg-slate-950 border border-slate-800 rounded-xl p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-emerald-400" />
                <span>Discrete Time Windows (Δt = 10s)</span>
              </h3>
              <span className="text-xs font-mono text-slate-400">
                {windows.length} States
              </span>
            </div>

            <div className="space-y-2 max-h-[540px] overflow-y-auto pr-1">
              {windows.map((w) => {
                const isSelected = selectedWindow?.windowIndex === w.windowIndex;
                const isAttack = w.attackFraction > 0.3;

                return (
                  <button
                    key={w.windowIndex}
                    onClick={() => setSelectedWindowIdx(w.windowIndex)}
                    className={`w-full text-left p-3 rounded-lg border transition-all flex items-center justify-between ${
                      isSelected
                        ? 'bg-slate-900 border-emerald-500/70 shadow-md ring-1 ring-emerald-500/20'
                        : 'bg-slate-900/50 border-slate-800/80 hover:bg-slate-900 hover:border-slate-700'
                    }`}
                  >
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono font-semibold text-slate-200">
                          State s_{w.windowIndex}
                        </span>
                        <span className="text-[11px] text-slate-400 font-mono">
                          [{w.windowStartTime} → {w.windowEndTime}]
                        </span>
                        {isAttack && (
                          <span className="px-1.5 py-0.2 rounded text-[10px] font-semibold bg-red-500/20 text-red-300 border border-red-500/30">
                            {Math.round(w.attackFraction * 100)}% Attack
                          </span>
                        )}
                      </div>

                      <div className="grid grid-cols-3 gap-2 mt-2 text-[11px] text-slate-400 font-mono">
                        <div>
                          <span className="text-slate-500">Flows:</span>{' '}
                          <span className="text-slate-300 font-semibold">{w.totalConnections}</span>
                        </div>
                        <div>
                          <span className="text-slate-500">Flags:</span>{' '}
                          <span className="text-slate-300 font-semibold">{w.sumSynAckRstFlags}</span>
                        </div>
                        <div>
                          <span className="text-slate-500">Ports:</span>{' '}
                          <span className="text-slate-300 font-semibold">{w.uniqueDestPorts}</span>
                        </div>
                      </div>
                    </div>

                    <ArrowRight
                      className={`w-4 h-4 transition ${
                        isSelected ? 'text-emerald-400 translate-x-0.5' : 'text-slate-600'
                      }`}
                    />
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        {/* Right Column: Mathematical State Vector Inspector */}
        <div className="lg:col-span-7 space-y-4">
          {selectedWindow ? (
            <div className="bg-slate-950 border border-slate-800 rounded-xl p-5 space-y-5">
              {/* Header */}
              <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-slate-800 gap-2">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-base font-bold text-slate-100 font-mono">
                      State Vector s_{selectedWindow.windowIndex}
                    </span>
                    <span className="px-2 py-0.5 rounded text-xs font-mono bg-slate-800 text-slate-300">
                      {selectedWindow.windowStartTime} – {selectedWindow.windowEndTime}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">
                    Normalized 7-dimensional continuous feature representation for Cyber World Model state dynamics.
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <div className="px-3 py-1.5 rounded-lg bg-slate-900 border border-slate-800 text-right">
                    <span className="text-[10px] text-slate-400 block uppercase font-mono">Attack Fraction y_t</span>
                    <span className={`text-xs font-mono font-bold ${
                      selectedWindow.attackFraction > 0.3 ? 'text-red-400' : 'text-emerald-400'
                    }`}>
                      {(selectedWindow.attackFraction * 100).toFixed(1)}% ({selectedWindow.attackFraction.toFixed(3)})
                    </span>
                  </div>
                </div>
              </div>

              {/* State Vector Table (Raw vs Normalized) */}
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2.5">
                  Dynamic State Vector Components (X vs Scaled Z)
                </h4>
                <div className="border border-slate-800 rounded-lg overflow-hidden">
                  <table className="w-full text-left text-xs font-mono">
                    <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800">
                      <tr>
                        <th className="py-2 px-3 font-medium">Feature Dimension</th>
                        <th className="py-2 px-3 font-medium">Raw Value (X)</th>
                        <th className="py-2 px-3 font-medium">StandardScaler (Z)</th>
                        <th className="py-2 px-3 font-medium">Interpretation</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 bg-slate-950">
                      <tr>
                        <td className="py-2 px-3 text-slate-300 font-semibold">1. Total Connections</td>
                        <td className="py-2 px-3 text-slate-400">{selectedWindow.totalConnections}</td>
                        <td className="py-2 px-3 text-emerald-400 font-semibold">
                          {selectedWindow.normTotalConnections?.toFixed(3)}
                        </td>
                        <td className="py-2 px-3 text-slate-500 text-[11px]">Flow volume density</td>
                      </tr>
                      <tr>
                        <td className="py-2 px-3 text-slate-300 font-semibold">2. Sum SYN/ACK/RST Flags</td>
                        <td className="py-2 px-3 text-slate-400">{selectedWindow.sumSynAckRstFlags}</td>
                        <td className="py-2 px-3 text-emerald-400 font-semibold">
                          {selectedWindow.normSumFlags?.toFixed(3)}
                        </td>
                        <td className="py-2 px-3 text-slate-500 text-[11px]">Protocol state friction / floods</td>
                      </tr>
                      <tr>
                        <td className="py-2 px-3 text-slate-300 font-semibold">3. Unique Destination Ports</td>
                        <td className="py-2 px-3 text-slate-400">{selectedWindow.uniqueDestPorts}</td>
                        <td className="py-2 px-3 text-emerald-400 font-semibold">
                          {selectedWindow.normUniquePorts?.toFixed(3)}
                        </td>
                        <td className="py-2 px-3 text-slate-500 text-[11px]">Horizontal scan dispersion</td>
                      </tr>
                      <tr>
                        <td className="py-2 px-3 text-slate-300 font-semibold">4. Average Bytes/s</td>
                        <td className="py-2 px-3 text-slate-400">{selectedWindow.avgBytesPerSec.toLocaleString()}</td>
                        <td className="py-2 px-3 text-emerald-400 font-semibold">
                          {selectedWindow.normAvgBytes?.toFixed(3)}
                        </td>
                        <td className="py-2 px-3 text-slate-500 text-[11px]">Network payload bandwidth load</td>
                      </tr>
                      <tr>
                        <td className="py-2 px-3 text-slate-300 font-semibold">5. Average Packets/s</td>
                        <td className="py-2 px-3 text-slate-400">{selectedWindow.avgPacketsPerSec.toLocaleString()}</td>
                        <td className="py-2 px-3 text-emerald-400 font-semibold">
                          {selectedWindow.normAvgPackets?.toFixed(3)}
                        </td>
                        <td className="py-2 px-3 text-slate-500 text-[11px]">Packet transmission velocity</td>
                      </tr>
                      <tr>
                        <td className="py-2 px-3 text-slate-300 font-semibold">6. Average IAT Mean (μ)</td>
                        <td className="py-2 px-3 text-slate-400">{selectedWindow.avgIatMean.toLocaleString()} μs</td>
                        <td className="py-2 px-3 text-emerald-400 font-semibold">
                          {selectedWindow.normAvgIatMean?.toFixed(3)}
                        </td>
                        <td className="py-2 px-3 text-slate-500 text-[11px]">Packet inter-arrival rhythm</td>
                      </tr>
                      <tr>
                        <td className="py-2 px-3 text-slate-300 font-semibold">7. Average IAT Variance (σ²)</td>
                        <td className="py-2 px-3 text-slate-400">{selectedWindow.avgIatVariance.toLocaleString()}</td>
                        <td className="py-2 px-3 text-emerald-400 font-semibold">
                          {selectedWindow.normAvgIatVariance?.toFixed(3)}
                        </td>
                        <td className="py-2 px-3 text-slate-500 text-[11px]">Network jitter / timing chaos</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Vector Representation Preview */}
              <div className="p-3.5 rounded-lg bg-slate-900 border border-slate-800">
                <span className="text-xs font-mono text-slate-400 block mb-1">
                  Numpy Tensor Input (1, 7):
                </span>
                <code className="text-xs font-mono text-emerald-300 block bg-slate-950 p-2.5 rounded border border-slate-800/80 overflow-x-auto">
                  s_{selectedWindow.windowIndex} = np.array([
                  {selectedWindow.normTotalConnections?.toFixed(2)},{' '}
                  {selectedWindow.normSumFlags?.toFixed(2)},{' '}
                  {selectedWindow.normUniquePorts?.toFixed(2)},{' '}
                  {selectedWindow.normAvgBytes?.toFixed(2)},{' '}
                  {selectedWindow.normAvgPackets?.toFixed(2)},{' '}
                  {selectedWindow.normAvgIatMean?.toFixed(2)},{' '}
                  {selectedWindow.normAvgIatVariance?.toFixed(2)}
                  ])
                </code>
              </div>

              {/* Underlying Flows in Window */}
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">
                  Underlying Raw Flows in Window ({selectedWindowFlows.length} flows)
                </h4>
                <div className="max-h-48 overflow-y-auto border border-slate-800 rounded-lg text-[11px] font-mono">
                  <table className="w-full text-left">
                    <thead className="bg-slate-900 text-slate-400 sticky top-0">
                      <tr>
                        <th className="py-1.5 px-2.5">Time</th>
                        <th className="py-1.5 px-2.5">Dst Port</th>
                        <th className="py-1.5 px-2.5">SYN/ACK/RST</th>
                        <th className="py-1.5 px-2.5">Bytes/s</th>
                        <th className="py-1.5 px-2.5">Label</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/50 bg-slate-950/60">
                      {selectedWindowFlows.map((f) => (
                        <tr key={f.id} className="hover:bg-slate-900/40">
                          <td className="py-1 px-2.5 text-slate-400">{f.timestamp.substring(11)}</td>
                          <td className="py-1 px-2.5 text-slate-300">{f.destPort}</td>
                          <td className="py-1 px-2.5 text-slate-300">
                            {f.synFlagCount} / {f.ackFlagCount} / {f.rstFlagCount}
                          </td>
                          <td className="py-1 px-2.5 text-slate-300">{f.flowBytesPerSec.toLocaleString()}</td>
                          <td className="py-1 px-2.5">
                            <span
                              className={`px-1.5 py-0.2 rounded text-[10px] font-medium ${
                                f.label === 'BENIGN'
                                  ? 'bg-emerald-500/10 text-emerald-400'
                                  : 'bg-red-500/10 text-red-400 border border-red-500/30'
                              }`}
                            >
                              {f.label}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            <div className="bg-slate-950 border border-slate-800 rounded-xl p-8 text-center text-slate-400 text-sm">
              No time windows available. Click Reset or add flows.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
