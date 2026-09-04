export interface NetworkFlow {
  id: string;
  timestamp: string; // ISO or formatted
  destPort: number;
  flowDuration: number; // in microseconds
  totalFwdPackets: number;
  totalBackwardPackets: number;
  flowBytesPerSec: number;
  flowPacketsPerSec: number;
  synFlagCount: number;
  ackFlagCount: number;
  rstFlagCount: number;
  fwdIatMean: number;
  fwdIatStd: number;
  label: 'BENIGN' | 'PortScan' | 'DDoS' | 'DoS Hulk' | 'FTP-Patator';
}

export interface StateVectorWindow {
  windowIndex: number;
  windowStartTime: string;
  windowEndTime: string;
  totalConnections: number;
  sumSynAckRstFlags: number;
  uniqueDestPorts: number;
  avgBytesPerSec: number;
  avgPacketsPerSec: number;
  avgIatMean: number;
  avgIatVariance: number;
  attackFraction: number; // 0.0 to 1.0 (supervision label)
  // Normalized values
  normTotalConnections?: number;
  normSumFlags?: number;
  normUniquePorts?: number;
  normAvgBytes?: number;
  normAvgPackets?: number;
  normAvgIatMean?: number;
  normAvgIatVariance?: number;
}

export interface ColumnSpec {
  name: string;
  ids2017Name: string;
  ids2018Alias: string;
  type: string;
  role: 'Feature' | 'Grouping' | 'Label';
  description: string;
}
