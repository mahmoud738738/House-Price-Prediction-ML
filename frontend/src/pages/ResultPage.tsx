import React, { useEffect } from 'react';
import { useLocation, useNavigate, Link } from 'react-router-dom';
import type { PredictionResultState } from '../types/prediction';
import { Home, ArrowLeft, CheckCircle, Tag, MapPin, Layers, Bath, Compass, Shield } from 'lucide-react';

export const ResultPage: React.FC = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const state = location.state as PredictionResultState | undefined;

  useEffect(() => {
    if (!state || !state.result) {
      navigate('/', { replace: true });
    }
  }, [state, navigate]);

  if (!state || !state.result) {
    return null;
  }

  const { input, result } = state;

  return (
    <div className="max-w-2xl mx-auto space-y-8 pt-4">
      {/* Return Link */}
      <Link
        to="/"
        className="inline-flex items-center space-x-2 text-sm font-medium text-slate-600 hover:text-blue-600 transition"
      >
        <ArrowLeft className="w-4 h-4" />
        <span>Back to Valuation Form</span>
      </Link>

      {/* Main Result Card */}
      <div className="bg-white rounded-3xl shadow-xl border border-slate-100 overflow-hidden">
        <div className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 p-8 text-white text-center relative">
          <div className="inline-flex items-center space-x-1.5 bg-white/20 backdrop-blur-md px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider mb-3">
            <CheckCircle className="w-4 h-4 text-emerald-300" />
            <span>Predicted Market Price</span>
          </div>

          <h2 className="text-4xl sm:text-5xl font-black tracking-tight mb-2">
            {result.formatted_price}
          </h2>

          <p className="text-blue-100 text-sm">
            Approx. ₹ {result.predicted_price.toLocaleString('en-IN')} INR
          </p>

          <div className="mt-5 flex justify-center gap-6 text-xs text-blue-100/90 border-t border-white/10 pt-4">
            <div>
              <span className="block text-white font-bold text-base">{result.price_lac}</span>
              <span>Lacs (₹)</span>
            </div>
            <div className="border-r border-white/10"></div>
            <div>
              <span className="block text-white font-bold text-base">{result.price_cr}</span>
              <span>Crores (₹)</span>
            </div>
          </div>
        </div>

        {/* Input Parameters Summary */}
        <div className="p-6 sm:p-8 space-y-6">
          <h3 className="text-base font-bold text-slate-800 flex items-center space-x-2">
            <Tag className="w-4 h-4 text-blue-600" />
            <span>Evaluated Property Specifications</span>
          </h3>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3.5 text-sm">
            <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
              <span className="text-xs text-slate-400 block flex items-center space-x-1">
                <MapPin className="w-3.5 h-3.5 text-slate-400 inline" />
                <span>Location</span>
              </span>
              <span className="font-semibold text-slate-800 capitalize mt-0.5 block truncate">
                {input.location}
              </span>
            </div>

            <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
              <span className="text-xs text-slate-400 block flex items-center space-x-1">
                <Home className="w-3.5 h-3.5 text-slate-400 inline" />
                <span>Carpet Area</span>
              </span>
              <span className="font-semibold text-slate-800 mt-0.5 block">
                {input.carpet_area_sqft} sqft
              </span>
            </div>

            <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
              <span className="text-xs text-slate-400 block flex items-center space-x-1">
                <Layers className="w-3.5 h-3.5 text-slate-400 inline" />
                <span>Floor Level</span>
              </span>
              <span className="font-semibold text-slate-800 mt-0.5 block">
                {input.floor_num === 0 ? 'Ground' : input.floor_num === -1 ? 'Basement' : `Floor ${input.floor_num}`}
              </span>
            </div>

            <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
              <span className="text-xs text-slate-400 block flex items-center space-x-1">
                <Bath className="w-3.5 h-3.5 text-slate-400 inline" />
                <span>Bath & Balcony</span>
              </span>
              <span className="font-semibold text-slate-800 mt-0.5 block">
                {input.bathroom} Bath / {input.balcony} Balc
              </span>
            </div>

            <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
              <span className="text-xs text-slate-400 block flex items-center space-x-1">
                <Shield className="w-3.5 h-3.5 text-slate-400 inline" />
                <span>Furnishing</span>
              </span>
              <span className="font-semibold text-slate-800 mt-0.5 block">
                {input.furnishing}
              </span>
            </div>

            <div className="p-3 bg-slate-50 rounded-xl border border-slate-100">
              <span className="text-xs text-slate-400 block flex items-center space-x-1">
                <Compass className="w-3.5 h-3.5 text-slate-400 inline" />
                <span>Facing / Type</span>
              </span>
              <span className="font-semibold text-slate-800 mt-0.5 block">
                {input.facing} ({input.transaction})
              </span>
            </div>
          </div>

          <div className="pt-2">
            <button
              onClick={() => navigate('/')}
              className="w-full py-3.5 px-6 bg-slate-900 hover:bg-slate-800 text-white font-semibold rounded-xl shadow-md transition text-center cursor-pointer"
            >
              Evaluate Another Property
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
