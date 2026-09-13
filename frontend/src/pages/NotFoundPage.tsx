import React from 'react';
import { Link } from 'react-router-dom';
import { Home } from 'lucide-react';

export const NotFoundPage: React.FC = () => {
  return (
    <div className="min-h-[60vh] flex flex-col items-center justify-center text-center px-4">
      <h1 className="text-7xl font-extrabold text-slate-300">404</h1>
      <h2 className="text-2xl font-bold text-slate-800 mt-2">Page Not Found</h2>
      <p className="text-slate-500 mt-2 mb-6">The page you are looking for does not exist.</p>
      <Link
        to="/"
        className="inline-flex items-center space-x-2 px-5 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-xl shadow-md transition"
      >
        <Home className="w-4 h-4" />
        <span>Return to Home</span>
      </Link>
    </div>
  );
};
