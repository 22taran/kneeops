import React from 'react';
import { Activity, Settings } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="bg-white border-b border-medical-200 px-6 py-4">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-primary-600 rounded-lg flex items-center justify-center">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-medical-900">KneeOps</h1>
            <p className="text-xs text-medical-600">AI-Powered ACL Diagnosis</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          <button className="p-2 text-medical-600 hover:text-medical-900 hover:bg-medical-100 rounded-lg transition-colors">
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header; 