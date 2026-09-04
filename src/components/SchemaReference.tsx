import React from 'react';
import { EXPECTED_COLUMN_SPECS } from '../data/mockData';
import { AlertCircle, CheckCircle, Database, HelpCircle, FileSpreadsheet } from 'lucide-react';

export const SchemaReference: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Overview Banner */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5">
        <div className="flex items-center gap-2 mb-2">
          <Database className="w-4 h-4 text-emerald-400" />
          <h2 className="text-sm font-semibold text-slate-100">
            CIC-IDS2017 & CIC-IDS2018 Expected Schema Reference
          </h2>
        </div>
        <p className="text-xs text-slate-400 leading-relaxed max-w-3xl">
          Below are the exact assumed column names for the input file (<code className="text-emerald-300">data/traffic.csv</code>).
          The script applies automated header whitespace stripping (<code className="text-emerald-300">df.columns.str.strip()</code>)
          and reconciles CIC-IDS2018 abbreviated names so that both datasets work seamlessly without manual edits.
        </p>
      </div>

      {/* Dataset Quirks & Clean Handling Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1.5">
          <div className="flex items-center gap-2 text-amber-400 font-semibold">
            <AlertCircle className="w-4 h-4" />
            <span>Leading Header Whitespace</span>
          </div>
          <p className="text-slate-400 leading-relaxed">
            Raw CIC-IDS2017 CSV files from University of New Brunswick (UNB) contain accidental leading spaces in columns such as <code className="text-slate-300">" Destination Port"</code>. The script automatically strips all whitespace.
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1.5">
          <div className="flex items-center gap-2 text-amber-400 font-semibold">
            <AlertCircle className="w-4 h-4" />
            <span>Zero-Duration Division by Zero</span>
          </div>
          <p className="text-slate-400 leading-relaxed">
            Network flows with duration = 0 cause <code className="text-amber-300">bytes/0 = +inf</code> in <code className="text-slate-300">Flow Bytes/s</code>. The script coerces infinite values to NaNs and purges corrupted rows.
          </p>
        </div>

        <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-1.5">
          <div className="flex items-center gap-2 text-emerald-400 font-semibold">
            <CheckCircle className="w-4 h-4" />
            <span>Strict Label-Feature Separation</span>
          </div>
          <p className="text-slate-400 leading-relaxed">
            The <code className="text-slate-300">Label</code> column is kept strictly isolated from the 7 dynamic state vector features so that ground truth never leaks into unsupervised environment transitions.
          </p>
        </div>
      </div>

      {/* Full Schema Table */}
      <div className="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden shadow-lg">
        <div className="p-4 bg-slate-900/90 border-b border-slate-800 flex items-center justify-between">
          <h3 className="text-xs font-semibold text-slate-200 uppercase tracking-wider flex items-center gap-2">
            <FileSpreadsheet className="w-4 h-4 text-emerald-400" />
            <span>13 Selected Input Columns</span>
          </h3>
          <span className="text-xs text-slate-400 font-mono">11 Features + 1 Time + 1 Label</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-900/60 text-slate-400 border-b border-slate-800">
              <tr>
                <th className="py-2.5 px-4 font-medium">Expected Column Name</th>
                <th className="py-2.5 px-4 font-medium">CIC-IDS2017 Format</th>
                <th className="py-2.5 px-4 font-medium">CIC-IDS2018 Alias</th>
                <th className="py-2.5 px-4 font-medium">Data Type</th>
                <th className="py-2.5 px-4 font-medium">Role</th>
                <th className="py-2.5 px-4 font-medium">Description & Purpose</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 text-slate-300">
              {EXPECTED_COLUMN_SPECS.map((col) => (
                <tr key={col.name} className="hover:bg-slate-900/40">
                  <td className="py-2.5 px-4 font-bold text-emerald-400">{col.name}</td>
                  <td className="py-2.5 px-4 text-slate-400">{col.ids2017Name}</td>
                  <td className="py-2.5 px-4 text-slate-400">{col.ids2018Alias}</td>
                  <td className="py-2.5 px-4 text-slate-400">{col.type}</td>
                  <td className="py-2.5 px-4">
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                        col.role === 'Feature'
                          ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                          : col.role === 'Grouping'
                          ? 'bg-blue-500/10 text-blue-400 border border-blue-500/20'
                          : 'bg-purple-500/10 text-purple-400 border border-purple-500/20'
                      }`}
                    >
                      {col.role}
                    </span>
                  </td>
                  <td className="py-2.5 px-4 text-slate-400 text-[11px] font-sans">
                    {col.description}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
