import React from 'react';

export const Navbar: React.FC = () => {
  return (
    <header className="border-b border-[#EAEAE5] bg-white/90 backdrop-blur-sm sticky top-0 z-40">
      <div className="max-w-5xl mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-16">
          {/* Brand Identity */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded bg-[#18181B] flex items-center justify-center text-white text-xs font-semibold tracking-wider">
              CW
            </div>
            <div className="flex items-center gap-3">
              <span className="font-semibold text-[#18181B] tracking-tight text-base">
                Cyber World Model
              </span>
              <span className="hidden sm:inline-flex px-2 py-0.5 rounded text-[11px] font-mono font-medium bg-[#F4F4F2] text-[#52525B] border border-[#EAEAE5]">
                PyTorch LSTM
              </span>
            </div>
          </div>

          {/* Single Tab: Attack Forecast */}
          <nav className="flex items-center">
            <div
              id="tab-forecast-single"
              className="inline-flex items-center px-3 py-1.5 rounded text-xs font-semibold bg-[#18181B] text-white"
            >
              Attack Forecast
            </div>
          </nav>
        </div>
      </div>
    </header>
  );
};
