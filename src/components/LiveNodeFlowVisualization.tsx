import React, { useMemo } from 'react';
import { Radio } from 'lucide-react';
import type { ActiveScenario } from './ForecasterSimulator';

export type TopologyPattern = 'fan_out' | 'convergence' | 'single_pulse' | 'calm';

export interface LiveNodeFlowVisualizationProps {
  scenario: ActiveScenario;
  peakProb: number;
  isForecasting?: boolean;
  totalConnections?: number;
  synFlags?: number;
  destPorts?: number;
  bytesPerSec?: number;
  packetsPerSec?: number;
  iatMean?: number;
  isCustomTelemetry?: boolean;
}

/**
 * Resolves the visual topology pattern based on underlying feature values
 * ensuring that manual slider adjustments, CSV uploads, and presets
 * trigger the exact same visual representation.
 */
export function resolveTopologyPattern(params: {
  destPorts: number;
  packetsPerSec: number;
  bytesPerSec: number;
  totalConnections: number;
  synFlags: number;
  peakProb: number;
  scenario?: ActiveScenario;
  isCustomTelemetry?: boolean;
}): TopologyPattern {
  const {
    destPorts,
    packetsPerSec,
    bytesPerSec,
    totalConnections,
    synFlags,
    peakProb,
    scenario,
    isCustomTelemetry,
  } = params;

  // 1. High unique destination ports definitively triggers the FAN-OUT visualization
  // regardless of whether it came from a preset, CSV upload, or manual slider
  if (destPorts >= 4 || (destPorts >= 2 && destPorts >= totalConnections * 0.25 && peakProb >= 0.15)) {
    return 'fan_out';
  }

  // 2. High packet rate or byte throughput concentrated against target triggers CONVERGENCE flood (DoS/DDoS)
  if (packetsPerSec >= 600 || bytesPerSec >= 150000 || (totalConnections >= 30 && destPorts <= 3 && peakProb >= 0.25)) {
    return 'convergence';
  }

  // 3. Elevated flag count or repeated auth attempts on single port triggers SINGLE_PULSE (Patator / Brute Force)
  if ((synFlags >= 250 || (totalConnections >= 15 && peakProb >= 0.25)) && destPorts <= 3) {
    return 'single_pulse';
  }

  // 4. Model detects elevated risk: choose best matching visual signature
  if (peakProb >= 0.35) {
    if (destPorts > 1) return 'fan_out';
    if (packetsPerSec > 200) return 'convergence';
    return 'single_pulse';
  }

  // 5. If using preset without manual customization, use calibrated preset identity as baseline
  if (!isCustomTelemetry && scenario && scenario !== 'custom_csv') {
    if (scenario === 'portscan') return 'fan_out';
    if (scenario === 'dos') return 'convergence';
    if (scenario === 'ftp_patator') return 'single_pulse';
    if (scenario === 'normal') return 'calm';
  }

  // 6. Calm nominal baseline
  return 'calm';
}

export const LiveNodeFlowVisualization: React.FC<LiveNodeFlowVisualizationProps> = ({
  scenario,
  peakProb,
  isForecasting = false,
  totalConnections = 1,
  synFlags = 0,
  destPorts = 1,
  bytesPerSec = 300,
  packetsPerSec = 5,
  iatMean = 1000,
  isCustomTelemetry = false,
}) => {
  // Feature-driven topology pattern resolver
  const activePattern = useMemo(() => {
    return resolveTopologyPattern({
      destPorts,
      packetsPerSec,
      bytesPerSec,
      totalConnections,
      synFlags,
      peakProb,
      scenario,
      isCustomTelemetry,
    });
  }, [
    destPorts,
    packetsPerSec,
    bytesPerSec,
    totalConnections,
    synFlags,
    peakProb,
    scenario,
    isCustomTelemetry,
  ]);

  // Determine color scheme and status text based on threat probability and active pattern
  const flowTheme = useMemo(() => {
    if (peakProb < 0.25 && activePattern === 'calm') {
      return {
        accent: '#059669', // Emerald
        badgeBg: 'bg-emerald-50',
        badgeText: 'text-emerald-700',
        badgeBorder: 'border-emerald-200',
        dot: 'bg-emerald-500',
        edgeNormal: '#D4D4D0',
        edgeActive: '#059669',
        statusLabel: 'Nominal Bilateral Flows',
      };
    }

    if (peakProb < 0.55) {
      let statusLabel = 'Reconnaissance Probe Pattern';
      if (activePattern === 'fan_out') statusLabel = 'Multi-Port Reconnaissance Sweep';
      else if (activePattern === 'convergence') statusLabel = 'High-Volume Ingress Surge';
      else if (activePattern === 'single_pulse') statusLabel = 'Elevated Authentication Cycle';
      else statusLabel = 'Elevated Baseline Variance';

      return {
        accent: '#D97706', // Amber
        badgeBg: 'bg-amber-50',
        badgeText: 'text-amber-800',
        badgeBorder: 'border-amber-200',
        dot: 'bg-amber-500',
        edgeNormal: '#D4D4D0',
        edgeActive: '#D97706',
        statusLabel,
      };
    }

    let statusLabel = 'Critical Threat Dynamics';
    if (activePattern === 'fan_out') statusLabel = 'High-Entropy Port Scan Fan-Out';
    else if (activePattern === 'convergence') statusLabel = 'Convergent Flood Ingress';
    else if (activePattern === 'single_pulse') statusLabel = 'Critical Brute-Force Spray';

    return {
      accent: '#E11D48', // Rose
      badgeBg: 'bg-rose-50',
      badgeText: 'text-rose-800',
      badgeBorder: 'border-rose-200',
      dot: 'bg-rose-500',
      edgeNormal: '#D4D4D0',
      edgeActive: '#E11D48',
      statusLabel,
    };
  }, [peakProb, activePattern]);

  // Dynamically sized Fan-Out Target Nodes adapting to the actual destPorts feature
  const displayedTargets = useMemo(() => {
    const count = Math.min(7, Math.max(2, Math.round(destPorts)));
    const allCandidateTargets = [
      { port: '21', service: 'FTP', status: 'Closed' },
      { port: '22', service: 'SSH', status: 'Filtered' },
      { port: '80', service: 'HTTP', status: 'Open' },
      { port: '139', service: 'SMB', status: 'Closed' },
      { port: '443', service: 'HTTPS', status: 'Open' },
      { port: '445', service: 'DS-SMB', status: 'Filtered' },
      { port: '3389', service: 'RDP', status: 'Closed' },
    ];
    const selected = allCandidateTargets.slice(0, count);
    return selected.map((target, idx) => {
      const y = count === 1 ? 105 : 30 + (idx / (count - 1)) * 150;
      return { ...target, y };
    });
  }, [destPorts]);

  // DoS Inbound Distributed Nodes with rate distribution based on packetsPerSec
  const dosSources = useMemo(() => {
    const basePps = packetsPerSec > 0 ? packetsPerSec : 1200;
    return [
      { ip: '198.51.100.12', y: 32, rate: `${Math.round(basePps * 0.22)} pps` },
      { ip: '203.0.113.84', y: 68, rate: `${Math.round(basePps * 0.26)} pps` },
      { ip: '192.0.2.145', y: 105, rate: `${Math.round(basePps * 0.24)} pps` },
      { ip: '185.220.101.9', y: 142, rate: `${Math.round(basePps * 0.18)} pps` },
      { ip: '198.51.100.220', y: 178, rate: `${Math.round(basePps * 0.10)} pps` },
    ];
  }, [packetsPerSec]);

  return (
    <div
      id="live-node-flow-panel"
      className="rounded-lg bg-[#F4F4F2] p-5 border border-[#EAEAE5] space-y-4 shadow-xs transition-colors"
    >
      {/* Topology Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-3 border-b border-[#EAEAE5]">
        <div className="flex items-center gap-2.5">
          <Radio className="w-4 h-4 text-[#18181B] shrink-0" />
          <div>
            <div className="text-xs font-semibold uppercase tracking-wider text-[#18181B] flex items-center gap-2">
              <span>Active Session Topology &amp; Flow Dynamics</span>
              <span className={`w-1.5 h-1.5 rounded-full ${flowTheme.dot} animate-flow-pulse`} />
              {isCustomTelemetry && (
                <span className="font-mono text-[9px] uppercase px-1.5 py-0.5 rounded bg-white text-[#18181B] border border-[#EAEAE5] font-semibold">
                  Live Manual Input
                </span>
              )}
            </div>
            <p className="text-[11px] text-[#71717A] mt-0.5">
              Live edge stream evaluated from observation vector{' '}
              <code className="font-mono text-[10px] bg-white px-1 py-0.5 rounded border border-[#EAEAE5]">
                t
              </code>
            </p>
          </div>
        </div>

        {/* Live Flow Pattern Status Badge */}
        <div className="flex items-center gap-2 self-start sm:self-auto">
          <div
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-[11px] font-semibold border ${flowTheme.badgeBg} ${flowTheme.badgeText} ${flowTheme.badgeBorder}`}
          >
            <span className={`w-1.5 h-1.5 rounded-full ${flowTheme.dot}`} />
            <span>{flowTheme.statusLabel}</span>
          </div>
          <span className="font-mono text-[11px] text-[#71717A] hidden md:inline">
            {destPorts > 1 ? `${destPorts} dst ports` : `${totalConnections} conns`} &bull;{' '}
            {packetsPerSec.toLocaleString(undefined, { maximumFractionDigits: 0 })} pkt/s
          </span>
        </div>
      </div>

      {/* SVG Canvas for Topology Graph */}
      <div className="w-full h-52 sm:h-56 bg-white rounded border border-[#EAEAE5] overflow-hidden relative select-none">
        <svg
          viewBox="0 0 760 210"
          className="w-full h-full"
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            {/* Gradient definition for active edges */}
            <linearGradient id="anomalousEdgeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#18181B" stopOpacity="0.8" />
              <stop offset="50%" stopColor={flowTheme.accent} stopOpacity="1" />
              <stop offset="100%" stopColor={flowTheme.accent} stopOpacity="0.9" />
            </linearGradient>

            <linearGradient id="normalEdgeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#71717A" stopOpacity="0.4" />
              <stop offset="100%" stopColor="#71717A" stopOpacity="0.7" />
            </linearGradient>
          </defs>

          {/* Background subtle grid guides */}
          <line x1="20" y1="105" x2="740" y2="105" stroke="#F4F4F2" strokeWidth="1" strokeDasharray="4 4" />
          <line x1="380" y1="15" x2="380" y2="195" stroke="#F4F4F2" strokeWidth="1" strokeDasharray="4 4" />

          {/* ============================================================ */}
          {/* 1. FAN-OUT PATTERN (HIGH UNIQUE DEST PORTS)                 */}
          {/* ============================================================ */}
          {activePattern === 'fan_out' && (
            <g id="topology-portscan-view">
              {/* Fan-Out Edges from Scanner (175, 105) to all dynamic target ports */}
              {displayedTargets.map((target, idx) => {
                const targetX = 610;
                const pathD = `M 175 105 C 320 105, 450 ${target.y}, ${targetX - 5} ${target.y}`;
                const dur = `${(0.8 + idx * 0.08).toFixed(2)}s`;
                const begin = `${(idx * 0.12).toFixed(2)}s`;

                return (
                  <g key={target.port}>
                    {/* Underlying static edge */}
                    <path
                      d={pathD}
                      fill="none"
                      stroke={flowTheme.accent}
                      strokeWidth="1.2"
                      strokeOpacity="0.35"
                    />

                    {/* Pulsing dashed edge indicating rapid SYN scan */}
                    <path
                      d={pathD}
                      fill="none"
                      stroke={flowTheme.accent}
                      strokeWidth="1.75"
                      strokeDasharray="5 5"
                      className="animate-flow-dash-fast"
                      strokeOpacity="0.85"
                    />

                    {/* Traveling SYN packet particle along path */}
                    <circle r="3" fill={flowTheme.accent}>
                      <animateMotion
                        path={pathD}
                        dur={dur}
                        repeatCount="indefinite"
                        begin={begin}
                      />
                    </circle>
                  </g>
                );
              })}

              {/* Source Node: Scanner Host */}
              <g transform="translate(45, 78)">
                <rect
                  x="0"
                  y="0"
                  width="130"
                  height="54"
                  rx="6"
                  fill="#18181B"
                  stroke={flowTheme.accent}
                  strokeWidth="1.5"
                />
                <circle cx="16" cy="18" r="4" fill={flowTheme.accent} className="animate-flow-pulse" />
                <text x="26" y="21" fill="#FFFFFF" fontSize="10" fontWeight="600" fontFamily="IBM Plex Sans">
                  SCANNER HOST
                </text>
                <text x="12" y="38" fill="#E4E4E7" fontSize="11" fontWeight="600" fontFamily="IBM Plex Mono">
                  192.168.10.14
                </text>
                <text x="12" y="49" fill="#A1A1AA" fontSize="8" fontFamily="IBM Plex Mono">
                  {destPorts} dst ports &bull; {packetsPerSec} pkt/s
                </text>
              </g>

              {/* Midpoint Fan-out Indicator Callout */}
              <g transform="translate(305, 92)">
                <rect
                  x="0"
                  y="0"
                  width="150"
                  height="26"
                  rx="4"
                  fill="#FAFAF8"
                  stroke="#EAEAE5"
                  strokeWidth="1"
                />
                <text x="75" y="17" textAnchor="middle" fill={flowTheme.accent} fontSize="9" fontWeight="600" fontFamily="IBM Plex Mono">
                  &larr; FAN-OUT ({destPorts} PORTS) &rarr;
                </text>
              </g>

              {/* Dynamic Target Port Nodes (Right Column) */}
              {displayedTargets.map((target) => (
                <g key={target.port} transform={`translate(605, ${target.y - 10})`}>
                  <rect
                    x="0"
                    y="0"
                    width="115"
                    height="20"
                    rx="3"
                    fill="#FAFAF8"
                    stroke="#EAEAE5"
                    strokeWidth="1"
                  />
                  <circle cx="8" cy="10" r="2.5" fill={flowTheme.accent} />
                  <text x="16" y="14" fill="#18181B" fontSize="10" fontWeight="600" fontFamily="IBM Plex Mono">
                    :{target.port}
                  </text>
                  <text x="56" y="14" fill="#71717A" fontSize="9" fontFamily="IBM Plex Sans">
                    {target.service}
                  </text>
                  <text x="106" y="14" textAnchor="end" fill="#A1A1AA" fontSize="8" fontFamily="IBM Plex Mono">
                    {target.status}
                  </text>
                </g>
              ))}
            </g>
          )}

          {/* ============================================================ */}
          {/* 2. CONVERGENCE PATTERN (HIGH PACKET / TRAFFIC FLOOD)       */}
          {/* ============================================================ */}
          {activePattern === 'convergence' && (
            <g id="topology-dos-view">
              {/* Converging Ingress Edges from distributed sources to victim (570, 105) */}
              {dosSources.map((source, idx) => {
                const sourceX = 175;
                const pathD = `M ${sourceX} ${source.y} C 360 ${source.y}, 440 105, 570 105`;
                const dur = `${(0.45 + idx * 0.05).toFixed(2)}s`;
                const begin = `${(idx * 0.08).toFixed(2)}s`;

                return (
                  <g key={source.ip}>
                    <path
                      d={pathD}
                      fill="none"
                      stroke="#E11D48"
                      strokeWidth="2"
                      strokeDasharray="4 4"
                      className="animate-flow-dash-fast"
                      strokeOpacity="0.9"
                    />
                    <circle r="3.5" fill="#E11D48">
                      <animateMotion
                        path={pathD}
                        dur={dur}
                        repeatCount="indefinite"
                        begin={begin}
                      />
                    </circle>
                  </g>
                );
              })}

              {/* Distributed Sources Column on Left */}
              {dosSources.map((source) => (
                <g key={source.ip} transform={`translate(45, ${source.y - 11})`}>
                  <rect
                    x="0"
                    y="0"
                    width="130"
                    height="22"
                    rx="3"
                    fill="#18181B"
                    stroke="#E11D48"
                    strokeWidth="1"
                  />
                  <circle cx="8" cy="11" r="2.5" fill="#E11D48" />
                  <text x="16" y="15" fill="#FFFFFF" fontSize="9" fontWeight="600" fontFamily="IBM Plex Mono">
                    {source.ip}
                  </text>
                  <text x="122" y="15" textAnchor="end" fill="#FDA4AF" fontSize="8" fontFamily="IBM Plex Mono">
                    {source.rate}
                  </text>
                </g>
              ))}

              {/* Central Convergent Target (Victim Gateway) */}
              <g transform="translate(570, 75)">
                <rect
                  x="0"
                  y="0"
                  width="145"
                  height="60"
                  rx="6"
                  fill="#18181B"
                  stroke="#E11D48"
                  strokeWidth="2"
                />
                <circle cx="16" cy="18" r="4" fill="#E11D48" className="animate-flow-pulse" />
                <text x="26" y="22" fill="#FFFFFF" fontSize="10" fontWeight="600" fontFamily="IBM Plex Sans">
                  VICTIM GATEWAY
                </text>
                <text x="14" y="40" fill="#FDA4AF" fontSize="11" fontWeight="600" fontFamily="IBM Plex Mono">
                  192.168.10.50 :80
                </text>
                <text x="14" y="52" fill="#A1A1AA" fontSize="8" fontFamily="IBM Plex Mono">
                  Queue: {Math.min(99, Math.max(15, Math.round(peakProb * 100)))}% SATURATED
                </text>
              </g>

              {/* Ingress Stream Marker */}
              <g transform="translate(295, 92)">
                <rect
                  x="0"
                  y="0"
                  width="170"
                  height="26"
                  rx="4"
                  fill="#FFF1F2"
                  stroke="#FECDD3"
                  strokeWidth="1"
                />
                <text x="85" y="17" textAnchor="middle" fill="#E11D48" fontSize="9" fontWeight="600" fontFamily="IBM Plex Mono">
                  &gt;&gt; CONVERGING FLOOD ({totalConnections}c) &gt;&gt;
                </text>
              </g>
            </g>
          )}

          {/* ============================================================ */}
          {/* 3. SINGLE-PULSE PATTERN (AUTH SPRAY / HIGH FLAGS SINGLE PORT)*/}
          {/* ============================================================ */}
          {activePattern === 'single_pulse' && (
            <g id="topology-ftp-view">
              {/* Single heavy bidirectional connection between client and daemon */}
              <g>
                <line
                  x1="185"
                  y1="105"
                  x2="555"
                  y2="105"
                  stroke={flowTheme.accent}
                  strokeWidth="3.5"
                  strokeDasharray="6 6"
                  className="animate-flow-dash-fast"
                  strokeOpacity="0.85"
                />

                {/* Repeated attempt bursts */}
                <circle r="4.5" fill={flowTheme.accent}>
                  <animateMotion
                    path="M 185 105 L 555 105"
                    dur="0.6s"
                    repeatCount="indefinite"
                  />
                </circle>

                {/* Rejection / RST return packet */}
                <circle r="3" fill="#71717A">
                  <animateMotion
                    path="M 555 105 L 185 105"
                    dur="0.6s"
                    begin="0.3s"
                    repeatCount="indefinite"
                  />
                </circle>
              </g>

              {/* Source Attacker Host */}
              <g transform="translate(50, 75)">
                <rect
                  x="0"
                  y="0"
                  width="135"
                  height="60"
                  rx="6"
                  fill="#18181B"
                  stroke={flowTheme.accent}
                  strokeWidth="1.5"
                />
                <circle cx="16" cy="18" r="4" fill={flowTheme.accent} />
                <text x="26" y="22" fill="#FFFFFF" fontSize="10" fontWeight="600" fontFamily="IBM Plex Sans">
                  BRUTE-FORCE AGENT
                </text>
                <text x="14" y="42" fill="#FFFFFF" fontSize="11" fontWeight="600" fontFamily="IBM Plex Mono">
                  192.168.10.51
                </text>
                <text x="14" y="53" fill="#A1A1AA" fontSize="8" fontFamily="IBM Plex Mono">
                  {totalConnections} conns &bull; {destPorts} port
                </text>
              </g>

              {/* Middle Action Indicator */}
              <g transform="translate(265, 78)">
                <rect
                  x="0"
                  y="0"
                  width="210"
                  height="54"
                  rx="4"
                  fill="#FAFAF8"
                  stroke="#EAEAE5"
                  strokeWidth="1"
                />
                <text x="105" y="18" textAnchor="middle" fill={flowTheme.accent} fontSize="10" fontWeight="600" fontFamily="IBM Plex Mono">
                  TCP PORT 21 SPRAY ({synFlags} FLAGS)
                </text>
                <text x="105" y="33" textAnchor="middle" fill="#71717A" fontSize="9" fontFamily="IBM Plex Mono">
                  REQ: USER &rarr; 530 Login Inc.
                </text>
                <text x="105" y="46" textAnchor="middle" fill="#18181B" fontSize="9" fontWeight="600" fontFamily="IBM Plex Mono">
                  {Math.max(10, Math.round(packetsPerSec * 6))} attempts/min &bull; Rapid IAT
                </text>
              </g>

              {/* Target FTP Server */}
              <g transform="translate(555, 75)">
                <rect
                  x="0"
                  y="0"
                  width="145"
                  height="60"
                  rx="6"
                  fill="#18181B"
                  stroke="#EAEAE5"
                  strokeWidth="1"
                />
                <circle cx="16" cy="18" r="4" fill="#059669" />
                <text x="26" y="22" fill="#FFFFFF" fontSize="10" fontWeight="600" fontFamily="IBM Plex Sans">
                  FTP SERVER (vsftpd)
                </text>
                <text x="14" y="42" fill="#FFFFFF" fontSize="11" fontWeight="600" fontFamily="IBM Plex Mono">
                  192.168.10.50 :21
                </text>
                <text x="14" y="53" fill="#E11D48" fontSize="8" fontWeight="600" fontFamily="IBM Plex Mono">
                  AUTH FAILURE SPIKE
                </text>
              </g>
            </g>
          )}

          {/* ============================================================ */}
          {/* 4. CALM BASELINE PATTERN (STEADY BILATERAL MESH)            */}
          {/* ============================================================ */}
          {activePattern === 'calm' && (
            <g id="topology-normal-view">
              {/* Calm bilateral flow lines */}
              <g>
                <path
                  d="M 180 80 C 320 60, 420 60, 560 80"
                  fill="none"
                  stroke="#059669"
                  strokeWidth="1.5"
                  strokeDasharray="4 4"
                  className="animate-flow-dash"
                  strokeOpacity="0.7"
                />
                <circle r="3" fill="#059669">
                  <animateMotion
                    path="M 180 80 C 320 60, 420 60, 560 80"
                    dur="1.8s"
                    repeatCount="indefinite"
                  />
                </circle>

                <path
                  d="M 560 130 C 420 150, 320 150, 180 130"
                  fill="none"
                  stroke="#71717A"
                  strokeWidth="1.2"
                  strokeDasharray="4 4"
                  className="animate-flow-dash"
                  strokeOpacity="0.5"
                />
                <circle r="2.5" fill="#71717A">
                  <animateMotion
                    path="M 560 130 C 420 150, 320 150, 180 130"
                    dur="2.2s"
                    repeatCount="indefinite"
                  />
                </circle>
              </g>

              {/* Internal Workstation */}
              <g transform="translate(50, 75)">
                <rect
                  x="0"
                  y="0"
                  width="130"
                  height="60"
                  rx="6"
                  fill="#18181B"
                  stroke="#EAEAE5"
                  strokeWidth="1"
                />
                <circle cx="16" cy="18" r="4" fill="#059669" />
                <text x="26" y="22" fill="#FFFFFF" fontSize="10" fontWeight="600" fontFamily="IBM Plex Sans">
                  WORKSTATION
                </text>
                <text x="14" y="42" fill="#FFFFFF" fontSize="11" fontWeight="600" fontFamily="IBM Plex Mono">
                  192.168.10.12
                </text>
                <text x="14" y="53" fill="#A1A1AA" fontSize="8" fontFamily="IBM Plex Mono">
                  {totalConnections} nominal conns
                </text>
              </g>

              {/* Middle Mesh Status Indicator */}
              <g transform="translate(290, 85)">
                <rect
                  x="0"
                  y="0"
                  width="180"
                  height="40"
                  rx="4"
                  fill="#FAFAF8"
                  stroke="#EAEAE5"
                  strokeWidth="1"
                />
                <text x="90" y="18" textAnchor="middle" fill="#059669" fontSize="10" fontWeight="600" fontFamily="IBM Plex Mono">
                  STEADY-STATE SESSION
                </text>
                <text x="90" y="32" textAnchor="middle" fill="#71717A" fontSize="9" fontFamily="IBM Plex Mono">
                  {packetsPerSec} pkt/s &bull; {(bytesPerSec / 1024).toFixed(1)} KB/s
                </text>
              </g>

              {/* Enterprise Proxy / Cloud Destination */}
              <g transform="translate(560, 75)">
                <rect
                  x="0"
                  y="0"
                  width="150"
                  height="60"
                  rx="6"
                  fill="#18181B"
                  stroke="#EAEAE5"
                  strokeWidth="1"
                />
                <circle cx="16" cy="18" r="4" fill="#059669" />
                <text x="26" y="22" fill="#FFFFFF" fontSize="10" fontWeight="600" fontFamily="IBM Plex Sans">
                  GATEWAY PROXY
                </text>
                <text x="14" y="42" fill="#FFFFFF" fontSize="11" fontWeight="600" fontFamily="IBM Plex Mono">
                  142.250.190.46 :443
                </text>
                <text x="14" y="53" fill="#A1A1AA" fontSize="8" fontFamily="IBM Plex Mono">
                  TLS 1.3 &bull; Healthy
                </text>
              </g>
            </g>
          )}
        </svg>
      </div>

      {/* Glanceable Technical Monospace Telemetry Strip */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-1 text-[11px] font-mono text-[#71717A]">
        <div className="flex items-center gap-4 flex-wrap">
          <span>
            <strong className="text-[#18181B]">PATTERN:</strong>{' '}
            {activePattern === 'fan_out'
              ? `Rapid SYN Sweep (${destPorts} Unique Dst Ports Fan-Out)`
              : activePattern === 'convergence'
              ? `Multi-Source Convergent Ingress (${packetsPerSec.toLocaleString()} pkt/s Flood)`
              : activePattern === 'single_pulse'
              ? `Persistent Single-Edge Credential Spray (${synFlags} Flags)`
              : 'Balanced Bilateral Client-Server Sessions'}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <span>
            FLOWS: <strong className="text-[#18181B]">{destPorts > 1 ? destPorts : totalConnections}</strong>
          </span>
          <span>&bull;</span>
          <span>
            RATE: <strong className="text-[#18181B]">{(bytesPerSec / 1024).toFixed(1)} KB/s</strong>
          </span>
        </div>
      </div>
    </div>
  );
};
