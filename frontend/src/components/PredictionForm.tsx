import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import type { PredictionRequest } from '../types/prediction';
import { predictPrice } from '../api/predictionClient';
import locationsData from '../locations.json';
import { Building2, Sparkles, AlertCircle, Loader2 } from 'lucide-react';

interface FormErrors {
  location?: string;
  carpet_area_sqft?: string;
  floor_num?: string;
  bathroom?: string;
  balcony?: string;
}

export const PredictionForm: React.FC = () => {
  const navigate = useNavigate();

  const [formData, setFormData] = useState<PredictionRequest>({
    location: locationsData.includes('thane') ? 'thane' : locationsData[0] || 'other',
    carpet_area_sqft: 850,
    floor_num: 3,
    bathroom: 2,
    balcony: 1,
    furnishing: 'Semi-Furnished',
    transaction: 'Resale',
    ownership: 'Freehold',
    facing: 'East',
  });

  const [errors, setErrors] = useState<FormErrors>({});
  const [loading, setLoading] = useState<boolean>(false);
  const [apiError, setApiError] = useState<string | null>(null);

  const validate = (): boolean => {
    const newErrors: FormErrors = {};

    if (!formData.location || formData.location.trim() === '') {
      newErrors.location = 'Please choose a location.';
    }

    if (!formData.carpet_area_sqft || formData.carpet_area_sqft <= 0) {
      newErrors.carpet_area_sqft = 'Carpet area must be greater than 0 sqft.';
    } else if (formData.carpet_area_sqft > 20000) {
      newErrors.carpet_area_sqft = 'Carpet area must be 20,000 sqft or less.';
    }

    if (formData.floor_num === undefined || formData.floor_num < -2 || formData.floor_num > 150) {
      newErrors.floor_num = 'Floor must be between -2 (Basement) and 150.';
    }

    if (!formData.bathroom || formData.bathroom < 1) {
      newErrors.bathroom = 'At least 1 bathroom is required.';
    }

    if (formData.balcony === undefined || formData.balcony < 0) {
      newErrors.balcony = 'Balcony cannot be negative.';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setApiError(null);

    if (!validate()) {
      return;
    }

    setLoading(true);

    try {
      const response = await predictPrice(formData);
      navigate('/result', {
        state: {
          input: formData,
          result: response,
        },
      });
    } catch (err: any) {
      setApiError(err.message || 'Failed to generate valuation. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-2xl shadow-xl border border-slate-100 p-6 md:p-8 max-w-3xl mx-auto">
      <div className="flex items-center space-x-3 mb-6 border-b border-slate-100 pb-4">
        <div className="p-2.5 bg-blue-50 text-blue-600 rounded-xl">
          <Building2 className="w-6 h-6" />
        </div>
        <div>
          <h2 className="text-xl font-bold text-slate-800">Property Details</h2>
          <p className="text-sm text-slate-500">Provide property specifications for instant machine-learning valuation</p>
        </div>
      </div>

      {apiError && (
        <div className="mb-6 p-4 bg-rose-50 border border-rose-200 text-rose-700 rounded-xl flex items-start space-x-3 text-sm">
          <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold">Prediction Failed</p>
            <p>{apiError}</p>
          </div>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Row 1: Location & Carpet Area */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-1.5">
              Location / City <span className="text-rose-500">*</span>
            </label>
            <select
              value={formData.location}
              onChange={(e) => setFormData({ ...formData, location: e.target.value })}
              className={`w-full px-3.5 py-2.5 bg-slate-50 border rounded-xl text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 capitalize transition ${
                errors.location ? 'border-rose-400 bg-rose-50/30' : 'border-slate-200'
              }`}
            >
              {locationsData.map((loc) => (
                <option key={loc} value={loc}>
                  {loc.charAt(0).toUpperCase() + loc.slice(1)}
                </option>
              ))}
            </select>
            {errors.location && <p className="text-xs text-rose-500 mt-1">{errors.location}</p>}
          </div>

          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-1.5">
              Carpet Area (sqft) <span className="text-rose-500">*</span>
            </label>
            <input
              type="number"
              min="1"
              max="20000"
              step="1"
              value={formData.carpet_area_sqft}
              onChange={(e) => setFormData({ ...formData, carpet_area_sqft: parseFloat(e.target.value) || 0 })}
              placeholder="e.g. 850"
              className={`w-full px-3.5 py-2.5 bg-slate-50 border rounded-xl text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 transition ${
                errors.carpet_area_sqft ? 'border-rose-400 bg-rose-50/30' : 'border-slate-200'
              }`}
            />
            {errors.carpet_area_sqft && <p className="text-xs text-rose-500 mt-1">{errors.carpet_area_sqft}</p>}
          </div>
        </div>

        {/* Row 2: Floor, Bathrooms, Balconies */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-1.5">
              Floor Number <span className="text-rose-500">*</span>
            </label>
            <input
              type="number"
              min="-2"
              max="150"
              value={formData.floor_num}
              onChange={(e) => setFormData({ ...formData, floor_num: parseInt(e.target.value, 10) || 0 })}
              className={`w-full px-3.5 py-2.5 bg-slate-50 border rounded-xl text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 transition ${
                errors.floor_num ? 'border-rose-400 bg-rose-50/30' : 'border-slate-200'
              }`}
            />
            <span className="text-[11px] text-slate-400 block mt-0.5">0 = Ground, -1 = Basement</span>
            {errors.floor_num && <p className="text-xs text-rose-500 mt-1">{errors.floor_num}</p>}
          </div>

          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-1.5">
              Bathrooms <span className="text-rose-500">*</span>
            </label>
            <input
              type="number"
              min="1"
              max="10"
              value={formData.bathroom}
              onChange={(e) => setFormData({ ...formData, bathroom: parseInt(e.target.value, 10) || 1 })}
              className={`w-full px-3.5 py-2.5 bg-slate-50 border rounded-xl text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 transition ${
                errors.bathroom ? 'border-rose-400 bg-rose-50/30' : 'border-slate-200'
              }`}
            />
            {errors.bathroom && <p className="text-xs text-rose-500 mt-1">{errors.bathroom}</p>}
          </div>

          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-1.5">
              Balconies
            </label>
            <input
              type="number"
              min="0"
              max="10"
              value={formData.balcony}
              onChange={(e) => setFormData({ ...formData, balcony: parseInt(e.target.value, 10) || 0 })}
              className={`w-full px-3.5 py-2.5 bg-slate-50 border rounded-xl text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 transition ${
                errors.balcony ? 'border-rose-400 bg-rose-50/30' : 'border-slate-200'
              }`}
            />
            {errors.balcony && <p className="text-xs text-rose-500 mt-1">{errors.balcony}</p>}
          </div>
        </div>

        {/* Row 3: Furnishing, Transaction, Ownership, Facing */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-1.5">Furnishing</label>
            <select
              value={formData.furnishing}
              onChange={(e) => setFormData({ ...formData, furnishing: e.target.value })}
              className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
            >
              <option value="Unfurnished">Unfurnished</option>
              <option value="Semi-Furnished">Semi-Furnished</option>
              <option value="Furnished">Furnished</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-1.5">Transaction Type</label>
            <select
              value={formData.transaction}
              onChange={(e) => setFormData({ ...formData, transaction: e.target.value })}
              className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
            >
              <option value="Resale">Resale</option>
              <option value="New Property">New Property</option>
            </select>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-1.5">Ownership</label>
            <select
              value={formData.ownership}
              onChange={(e) => setFormData({ ...formData, ownership: e.target.value })}
              className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
            >
              <option value="Freehold">Freehold</option>
              <option value="Co-operative Society">Co-operative Society</option>
              <option value="Leasehold">Leasehold</option>
              <option value="Power of Attorney">Power of Attorney</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-semibold text-slate-700 mb-1.5">Facing Direction</label>
            <select
              value={formData.facing}
              onChange={(e) => setFormData({ ...formData, facing: e.target.value })}
              className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-200 rounded-xl text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-500 transition"
            >
              <option value="East">East</option>
              <option value="North">North</option>
              <option value="North-East">North-East</option>
              <option value="West">West</option>
              <option value="South">South</option>
              <option value="North-West">North-West</option>
              <option value="South-East">South-East</option>
              <option value="South-West">South-West</option>
            </select>
          </div>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading}
          className="w-full py-3.5 px-6 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold rounded-xl shadow-lg shadow-blue-500/25 transition flex items-center justify-center space-x-2 disabled:opacity-60 cursor-pointer disabled:cursor-not-allowed"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Predicting Market Value...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5" />
              <span>Predict House Price</span>
            </>
          )}
        </button>
      </form>
    </div>
  );
};
