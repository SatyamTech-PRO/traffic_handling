/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { Navbar } from './components/Navbar';
import { ForecasterSimulator } from './components/ForecasterSimulator';

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-emerald-500/30 selection:text-emerald-200">
      <Navbar />

      <main className="flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 py-6 sm:py-8">
        <ForecasterSimulator />
      </main>

      <footer className="border-t border-slate-800/80 py-4 text-center text-xs text-slate-500">
        <div className="max-w-5xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>Cyber World Model • Autoregressive Threat Dynamics</span>
          <span className="font-mono text-[11px] text-slate-500">
            Real inference on held-out network sessions
          </span>
        </div>
      </footer>
    </div>
  );
}
