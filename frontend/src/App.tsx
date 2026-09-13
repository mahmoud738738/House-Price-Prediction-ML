import React from 'react';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import { HomePage } from './pages/HomePage';
import { ResultPage } from './pages/ResultPage';
import { NotFoundPage } from './pages/NotFoundPage';
import { Building2, ExternalLink } from 'lucide-react';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col font-sans">
        {/* Navigation Bar */}
        <header className="bg-white border-b border-slate-200 sticky top-0 z-50">
          <div className="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
            <Link to="/" className="flex items-center space-x-2.5">
              <div className="w-9 h-9 bg-blue-600 text-white rounded-xl flex items-center justify-center shadow-md shadow-blue-500/20">
                <Building2 className="w-5 h-5" />
              </div>
              <span className="font-bold text-lg text-slate-800 tracking-tight">
                Prophet<span className="text-blue-600">ML</span>
              </span>
            </Link>

            <div className="flex items-center space-x-4 text-sm font-medium">
              <Link to="/" className="text-slate-600 hover:text-blue-600 transition">
                Valuation Tool
              </Link>
              <a
                href="http://localhost:8000/docs"
                target="_blank"
                rel="noreferrer"
                className="hidden sm:flex items-center space-x-1 text-slate-600 hover:text-blue-600 transition"
              >
                <span>FastAPI Docs</span>
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
              <a
                href="https://github.com"
                target="_blank"
                rel="noreferrer"
                className="p-2 text-slate-500 hover:text-slate-800 transition"
                title="GitHub Repo"
              >
                <svg className="w-5 h-5 fill-current" viewBox="0 0 24 24">
                  <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
                </svg>
              </a>
            </div>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 py-8">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/result" element={<ResultPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Routes>
        </main>

        {/* Footer */}
        <footer className="border-t border-slate-200 bg-white py-6 text-center text-xs text-slate-500">
          <div className="max-w-6xl mx-auto px-4">
            <p>
              House Price Prediction (End-to-End ML Web App) • Trained on 171,000+ listings • Scikit-Learn Pipeline + FastAPI + React
            </p>
          </div>
        </footer>
      </div>
    </BrowserRouter>
  );
};

export default App;
