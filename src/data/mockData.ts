import { NetworkFlow, ColumnSpec } from '../types';
import demoScenariosRaw from './demo_scenarios.json';

export interface DemoForecastStep {
  step: number;
  predicted_state: number[];
  attack_probability: number;
  mitre_stage: string;
}

export interface DemoScenarioData {
  history: number[][];
  forecast_steps: DemoForecastStep[];
  top_shap_features: [string, number][];
  data_provenance: string;
}

export type ScenarioKey = 'normal' | 'portscan' | 'dos' | 'ftp_patator';

export const DEMO_SCENARIOS: Record<ScenarioKey, DemoScenarioData> = demoScenariosRaw as Record<ScenarioKey, DemoScenarioData>;

export const EXPECTED_COLUMN_SPECS: ColumnSpec[] = [
  {
    name: 'Timestamp',
    ids2017Name: 'Timestamp (often leading space: " Timestamp")',
    ids2018Alias: 'Timestamp',
    type: 'datetime',
    role: 'Grouping',
    description: 'Timestamp of packet capture used to bucket flows into discrete 10s temporal intervals.',
  },
  {
    name: 'Destination Port',
    ids2017Name: 'Destination Port (often: " Destination Port")',
    ids2018Alias: 'Dst Port',
    type: 'integer',
    role: 'Feature',
    description: 'Destination L4 port; used to compute unique destination port entropy per window.',
  },
  {
    name: 'Flow Duration',
    ids2017Name: 'Flow Duration',
    ids2018Alias: 'Flow Duration',
    type: 'numeric (μs)',
    role: 'Feature',
    description: 'Duration of the flow in microseconds; flows with 0 duration often trigger inf throughput.',
  },
  {
    name: 'Total Fwd Packets',
    ids2017Name: 'Total Fwd Packets',
    ids2018Alias: 'Tot Fwd Pkts',
    type: 'integer',
    role: 'Feature',
    description: 'Count of forward transmission packets.',
  },
  {
    name: 'Total Backward Packets',
    ids2017Name: 'Total Backward Packets',
    ids2018Alias: 'Tot Bwd Pkts',
    type: 'integer',
    role: 'Feature',
    description: 'Count of backward response packets.',
  },
  {
    name: 'Flow Bytes/s',
    ids2017Name: 'Flow Bytes/s',
    ids2018Alias: 'Flow Byts/s',
    type: 'float',
    role: 'Feature',
    description: 'Throughput in bytes per second; can contain +inf / -inf when duration is zero.',
  },
  {
    name: 'Flow Packets/s',
    ids2017Name: 'Flow Packets/s',
    ids2018Alias: 'Flow Pkts/s',
    type: 'float',
    role: 'Feature',
    description: 'Throughput in packets per second; cleaned for inf/NaN values.',
  },
  {
    name: 'SYN Flag Count',
    ids2017Name: 'SYN Flag Count',
    ids2018Alias: 'SYN Flag Cnt',
    type: 'integer',
    role: 'Feature',
    description: 'Packets carrying TCP SYN; key signal for SYN flood DoS and TCP half-open scans.',
  },
  {
    name: 'ACK Flag Count',
    ids2017Name: 'ACK Flag Count',
    ids2018Alias: 'ACK Flag Cnt',
    type: 'integer',
    role: 'Feature',
    description: 'Packets carrying TCP ACK; indicates established session traffic or ACK storms.',
  },
  {
    name: 'RST Flag Count',
    ids2017Name: 'RST Flag Count',
    ids2018Alias: 'RST Flag Cnt',
    type: 'integer',
    role: 'Feature',
    description: 'Packets carrying TCP RST; indicates closed connection rejections or tear-downs.',
  },
  {
    name: 'Fwd IAT Mean',
    ids2017Name: 'Fwd IAT Mean',
    ids2018Alias: 'Fwd IAT Mean',
    type: 'float',
    role: 'Feature',
    description: 'Mean inter-arrival time between packets sent in forward direction.',
  },
  {
    name: 'Fwd IAT Std',
    ids2017Name: 'Fwd IAT Std',
    ids2018Alias: 'Fwd IAT Std',
    type: 'float',
    role: 'Feature',
    description: 'Standard deviation of forward packet inter-arrival time; squared to yield IAT variance.',
  },
  {
    name: 'Label',
    ids2017Name: 'Label',
    ids2018Alias: 'Label',
    type: 'string',
    role: 'Label',
    description: 'Ground-truth classification ("BENIGN" vs attack). Isolated from state feature standardization.',
  },
];

export function generateSyntheticFlows(count: number = 60): NetworkFlow[] {
  const flows: NetworkFlow[] = [];
  const baseTime = new Date(2017, 6, 4, 9, 0, 0); // 2017-07-04 09:00:00

  for (let i = 0; i < count; i++) {
    // Spread across ~60 seconds (so we get ~6-8 windows of 10s each)
    const secOffset = (i / count) * 60 + (Math.random() * 2 - 1);
    const flowTime = new Date(baseTime.getTime() + Math.max(0, secOffset) * 1000);

    // Create 3 phases: 0-20s normal traffic, 20-40s PortScan / SYN Flood, 40-60s DDoS attack
    let label: NetworkFlow['label'] = 'BENIGN';
    if (secOffset >= 18 && secOffset <= 34) {
      label = Math.random() < 0.75 ? 'PortScan' : 'BENIGN';
    } else if (secOffset > 34 && secOffset <= 52) {
      label = Math.random() < 0.85 ? 'DDoS' : 'BENIGN';
    }

    let destPort = 80;
    let duration = 250000;
    let fwdPkts = 5;
    let bwdPkts = 4;
    let bytesSec = 3500;
    let pktsSec = 18;
    let syn = 0;
    let ack = 2;
    let rst = 0;
    let iatMean = 1200;
    let iatStd = 350;

    if (label === 'PortScan') {
      destPort = 1024 + Math.floor(Math.random() * 45000);
      duration = 800 + Math.floor(Math.random() * 3000);
      fwdPkts = 2;
      bwdPkts = 0;
      bytesSec = 450 + Math.random() * 300;
      pktsSec = 25 + Math.random() * 40;
      syn = 1;
      ack = 0;
      rst = Math.random() > 0.5 ? 1 : 0;
      iatMean = 120 + Math.random() * 150;
      iatStd = 25 + Math.random() * 40;
    } else if (label === 'DDoS') {
      destPort = Math.random() > 0.5 ? 80 : 443;
      duration = 1200000 + Math.floor(Math.random() * 500000);
      fwdPkts = 80 + Math.floor(Math.random() * 150);
      bwdPkts = 40 + Math.floor(Math.random() * 90);
      bytesSec = 120000 + Math.random() * 150000;
      pktsSec = 850 + Math.random() * 1200;
      syn = 12 + Math.floor(Math.random() * 15);
      ack = 45 + Math.floor(Math.random() * 60);
      rst = Math.floor(Math.random() * 3);
      iatMean = 45 + Math.random() * 120;
      iatStd = 15 + Math.random() * 30;
    } else {
      // Normal Benign
      const commonPorts = [80, 443, 22, 53, 8080, 3306];
      destPort = commonPorts[Math.floor(Math.random() * commonPorts.length)];
      duration = 20000 + Math.floor(Math.random() * 600000);
      fwdPkts = 3 + Math.floor(Math.random() * 20);
      bwdPkts = 2 + Math.floor(Math.random() * 15);
      bytesSec = 2000 + Math.random() * 18000;
      pktsSec = 10 + Math.random() * 60;
      syn = Math.random() > 0.7 ? 1 : 0;
      ack = 2 + Math.floor(Math.random() * 8);
      rst = 0;
      iatMean = 800 + Math.random() * 5000;
      iatStd = 200 + Math.random() * 1500;
    }

    flows.push({
      id: `flow-${i + 1}`,
      timestamp: flowTime.toISOString().replace('T', ' ').substring(0, 19),
      destPort,
      flowDuration: duration,
      totalFwdPackets: fwdPkts,
      totalBackwardPackets: bwdPkts,
      flowBytesPerSec: Math.round(bytesSec * 10) / 10,
      flowPacketsPerSec: Math.round(pktsSec * 10) / 10,
      synFlagCount: syn,
      ackFlagCount: ack,
      rstFlagCount: rst,
      fwdIatMean: Math.round(iatMean * 10) / 10,
      fwdIatStd: Math.round(iatStd * 10) / 10,
      label,
    });
  }

  return flows.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
}
