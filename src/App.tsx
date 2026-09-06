/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import { Navbar } from './components/Navbar';
import { ForecasterSimulator } from './components/ForecasterSimulator';

export default function App() {
  return (
    <div className="min-h-screen bg-[#FAFAF8] text-[#18181B] flex flex-col selection:bg-slate-200 selection:text-slate-900 antialiased">
      <Navbar />

      <main className="flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 py-12">
        <ForecasterSimulator />
      </main>

      <footer className="border-t border-[#EAEAE5] py-8 text-xs text-[#71717A]">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <span className="font-medium text-[#52525B]">Cyber World Model &bull; Autoregressive Threat Dynamics</span>
          <span className="font-mono text-[11px] text-[#A1A1AA]">
            Held-out network session evaluation &bull; 10s state resolution
          </span>
        </div>
      </footer>
    </div>
  );
}
