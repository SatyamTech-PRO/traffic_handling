import { NetworkFlow, StateVectorWindow } from '../types';

export function aggregateFlowsIntoWindows(
  flows: NetworkFlow[],
  windowSec: number = 10
): StateVectorWindow[] {
  if (!flows || flows.length === 0) return [];

  // Find min and max timestamps
  const timestamps = flows.map((f) => new Date(f.timestamp).getTime());
  const minTime = Math.min(...timestamps);
  const maxTime = Math.max(...timestamps);

  const windowMs = windowSec * 1000;
  const numWindows = Math.ceil((maxTime - minTime + 1) / windowMs);

  const rawWindows: StateVectorWindow[] = [];

  for (let w = 0; w < numWindows; w++) {
    const wStart = minTime + w * windowMs;
    const wEnd = wStart + windowMs;

    // Filter flows in this window
    const windowFlows = flows.filter((f) => {
      const t = new Date(f.timestamp).getTime();
      return t >= wStart && t < wEnd;
    });

    if (windowFlows.length === 0) continue;

    const totalConnections = windowFlows.length;

    // Sum of SYN + ACK + RST
    const sumSynAckRstFlags = windowFlows.reduce(
      (acc, f) => acc + f.synFlagCount + f.ackFlagCount + f.rstFlagCount,
      0
    );

    // Unique destination ports
    const uniquePorts = new Set(windowFlows.map((f) => f.destPort)).size;

    // Avg bytes/s and packets/s
    const avgBytesPerSec =
      windowFlows.reduce((acc, f) => acc + f.flowBytesPerSec, 0) / totalConnections;
    const avgPacketsPerSec =
      windowFlows.reduce((acc, f) => acc + f.flowPacketsPerSec, 0) / totalConnections;

    // Avg IAT Mean
    const avgIatMean =
      windowFlows.reduce((acc, f) => acc + f.fwdIatMean, 0) / totalConnections;

    // Avg IAT Variance (Fwd IAT Std ^ 2)
    const avgIatVariance =
      windowFlows.reduce((acc, f) => acc + Math.pow(f.fwdIatStd, 2), 0) / totalConnections;

    // Attack fraction
    const attackFlows = windowFlows.filter((f) => f.label !== 'BENIGN').length;
    const attackFraction = attackFlows / totalConnections;

    rawWindows.push({
      windowIndex: w + 1,
      windowStartTime: new Date(wStart).toTimeString().substring(0, 8),
      windowEndTime: new Date(wEnd).toTimeString().substring(0, 8),
      totalConnections,
      sumSynAckRstFlags,
      uniqueDestPorts: uniquePorts,
      avgBytesPerSec: Math.round(avgBytesPerSec * 100) / 100,
      avgPacketsPerSec: Math.round(avgPacketsPerSec * 100) / 100,
      avgIatMean: Math.round(avgIatMean * 100) / 100,
      avgIatVariance: Math.round(avgIatVariance * 100) / 100,
      attackFraction: Math.round(attackFraction * 1000) / 1000,
    });
  }

  // Apply StandardScaler normalization (Z = (X - mean) / std)
  if (rawWindows.length === 0) return [];

  const computeMeanStd = (vals: number[]) => {
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const variance =
      vals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / vals.length;
    const std = Math.sqrt(variance) || 1e-6; // prevent div by zero
    return { mean, std };
  };

  const statConn = computeMeanStd(rawWindows.map((w) => w.totalConnections));
  const statFlags = computeMeanStd(rawWindows.map((w) => w.sumSynAckRstFlags));
  const statPorts = computeMeanStd(rawWindows.map((w) => w.uniqueDestPorts));
  const statBytes = computeMeanStd(rawWindows.map((w) => w.avgBytesPerSec));
  const statPkts = computeMeanStd(rawWindows.map((w) => w.avgPacketsPerSec));
  const statIatMean = computeMeanStd(rawWindows.map((w) => w.avgIatMean));
  const statIatVar = computeMeanStd(rawWindows.map((w) => w.avgIatVariance));

  return rawWindows.map((w) => ({
    ...w,
    normTotalConnections: Math.round(((w.totalConnections - statConn.mean) / statConn.std) * 1000) / 1000,
    normSumFlags: Math.round(((w.sumSynAckRstFlags - statFlags.mean) / statFlags.std) * 1000) / 1000,
    normUniquePorts: Math.round(((w.uniqueDestPorts - statPorts.mean) / statPorts.std) * 1000) / 1000,
    normAvgBytes: Math.round(((w.avgBytesPerSec - statBytes.mean) / statBytes.std) * 1000) / 1000,
    normAvgPackets: Math.round(((w.avgPacketsPerSec - statPkts.mean) / statPkts.std) * 1000) / 1000,
    normAvgIatMean: Math.round(((w.avgIatMean - statIatMean.mean) / statIatMean.std) * 1000) / 1000,
    normAvgIatVariance: Math.round(((w.avgIatVariance - statIatVar.mean) / statIatVar.std) * 1000) / 1000,
  }));
}

export function generateCsvString(windows: StateVectorWindow[]): string {
  const headers = [
    'time_window',
    'total_connections',
    'sum_syn_ack_rst_flags',
    'unique_dest_ports',
    'avg_bytes_per_sec',
    'avg_packets_per_sec',
    'avg_iat_mean',
    'avg_iat_variance',
    'attack_fraction',
    'norm_total_connections',
    'norm_sum_syn_ack_rst_flags',
    'norm_unique_dest_ports',
    'norm_avg_bytes_per_sec',
    'norm_avg_packets_per_sec',
    'norm_avg_iat_mean',
    'norm_avg_iat_variance'
  ];

  const rows = windows.map((w) => [
    `${w.windowStartTime}-${w.windowEndTime}`,
    w.totalConnections,
    w.sumSynAckRstFlags,
    w.uniqueDestPorts,
    w.avgBytesPerSec,
    w.avgPacketsPerSec,
    w.avgIatMean,
    w.avgIatVariance,
    w.attackFraction,
    w.normTotalConnections ?? 0,
    w.normSumFlags ?? 0,
    w.normUniquePorts ?? 0,
    w.normAvgBytes ?? 0,
    w.normAvgPackets ?? 0,
    w.normAvgIatMean ?? 0,
    w.normAvgIatVariance ?? 0
  ]);

  return [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
}
