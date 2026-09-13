import React from 'react';
import { PredictionForm } from '../components/PredictionForm';
import { ShieldCheck, Cpu, Database, CheckCircle2 } from 'lucide-react';

export const HomePage: React.FC = () => {
  return (
    <div className="space-y-10">
      {/* Hero Header */}
      <div className="text-center max-w-2xl mx-auto pt-4">
        <div className="inline-flex items-center space-x-2 px-3.5 py-1.5 bg-blue-50 border border-blue-200 text-blue-700 text-xs font-semibold rounded-full mb-4">
          <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></span>
          <span>End-to-End Machine Learning Web App</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 tracking-tight">
          Indian House Price <span className="bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">Prediction</span>
        </h1>
        <p className="mt-3 text-base text-slate-600">
          Powered by an ensemble machine learning pipeline trained on 171,000+ real property transactions with minority sample-weighting for balanced valuation across all price brackets.
        </p>

        {/* Feature Badges */}
        <div className="flex flex-wrap items-center justify-center gap-4 mt-6 text-xs text-slate-600 font-medium">
          <div className="flex items-center space-x-1.5 bg-slate-100 px-3 py-1.5 rounded-lg">
            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
            <span>R² Score: 0.9259</span>
          </div>
          <div className="flex items-center space-x-1.5 bg-slate-100 px-3 py-1.5 rounded-lg">
            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
            <span>Balanced Sample Weights</span>
          </div>
          <div className="flex items-center space-x-1.5 bg-slate-100 px-3 py-1.5 rounded-lg">
            <CheckCircle2 className="w-4 h-4 text-emerald-600" />
            <span>Top-50 Indian Metros</span>
          </div>
        </div>
      </div>

      {/* Main Interactive Form */}
      <PredictionForm />

      {/* Highlights / Technical Architecture Footer */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto pt-4">
        <div className="p-5 bg-white rounded-xl border border-slate-200/80 shadow-sm text-center">
          <div className="w-10 h-10 bg-blue-50 text-blue-600 rounded-lg flex items-center justify-center mx-auto mb-3">
            <Database className="w-5 h-5" />
          </div>
          <h3 className="font-semibold text-slate-800 text-sm">Cleaned 187k Dataset</h3>
          <p className="text-xs text-slate-500 mt-1">
            Area units normalized to sqft, floor levels decoded, and high-frequency location grouping applied.
          </p>
        </div>

        <div className="p-5 bg-white rounded-xl border border-slate-200/80 shadow-sm text-center">
          <div className="w-10 h-10 bg-indigo-50 text-indigo-600 rounded-lg flex items-center justify-center mx-auto mb-3">
            <Cpu className="w-5 h-5" />
          </div>
          <h3 className="font-semibold text-slate-800 text-sm">Weighted RandomForest</h3>
          <p className="text-xs text-slate-500 mt-1">
            Minority price tiers weighted inversely to prevent under-prediction on luxury and budget listings.
          </p>
        </div>

        <div className="p-5 bg-white rounded-xl border border-slate-200/80 shadow-sm text-center">
          <div className="w-10 h-10 bg-emerald-50 text-emerald-600 rounded-lg flex items-center justify-center mx-auto mb-3">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <h3 className="font-semibold text-slate-800 text-sm">Production FastAPI</h3>
          <p className="text-xs text-slate-500 mt-1">
            Full scikit-learn Pipeline loaded once at startup via lifespan handler with sub-millisecond inference.
          </p>
        </div>
      </div>
    </div>
  );
};
