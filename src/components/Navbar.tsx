import React from 'react';
import { Shield, Crosshair } from 'lucide-react';

export const Navbar: React.FC = () => {
  return (
    <header className="border-b border-slate-800/80 bg-slate-950/80 backdrop-blur sticky top-0 z-40">
      <div className="max-w-5xl mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-16">
          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-400 shadow-sm shadow-emerald-950">
              <Shield className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-slate-100 tracking-tight text-sm sm:text-base">Cyber World Model</span>
                <span className="px-2 py-0.5 rounded-full text-[10px] font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  PyTorch Engine
                </span>
              </div>
              <p className="text-xs text-slate-400 hidden sm:block">
                Predictive Autoregressive Cyber Dynamics
              </p>
            </div>
          </div>

          {/* Single Tab: Attack Forecast */}
          <nav className="flex items-center">
            <div
              id="tab-forecast-single"
              className="flex items-center gap-2 px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-emerald-500 text-slate-950 shadow-sm"
            >
              <Crosshair className="w-3.5 h-3.5" />
              <span>Attack Forecast</span>
            </div>
          </nav>
        </div>
      </div>
    </header>
  );
};
