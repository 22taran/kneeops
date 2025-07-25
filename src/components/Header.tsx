import React from 'react';
import { Activity, Settings, Bell, HelpCircle } from 'lucide-react';
import ThemeToggle from './ThemeToggle';

interface HeaderProps {
  className?: string;
}

const Header: React.FC<HeaderProps> = ({ className = '' }) => {
  return (
    <header className={`bg-white dark:bg-medical-900 border-b border-medical-200 dark:border-medical-800 px-4 sm:px-6 py-3 shadow-sm ${className}`}>
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-primary-600 rounded-lg flex items-center justify-center shadow-md">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <div className="hidden sm:block">
            <h1 className="text-xl font-bold text-medical-900 dark:text-white">KneeOps</h1>
            <p className="text-xs text-medical-600 dark:text-medical-300">AI-Powered Knee MRI Analysis</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2 sm:space-x-4">
          <button 
            className="p-2 text-medical-600 dark:text-medical-300 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-medical-100 dark:hover:bg-medical-800 rounded-lg transition-colors duration-200"
            aria-label="Help"
          >
            <HelpCircle className="w-5 h-5" />
          </button>
          <button 
            className="p-2 text-medical-600 dark:text-medical-300 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-medical-100 dark:hover:bg-medical-800 rounded-lg transition-colors duration-200 relative"
            aria-label="Notifications"
          >
            <Bell className="w-5 h-5" />
            <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
          </button>
          <ThemeToggle />
          <div className="h-8 w-px bg-medical-200 dark:bg-medical-700 mx-1"></div>
          <button 
            className="p-2 text-medical-600 dark:text-medical-300 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-medical-100 dark:hover:bg-medical-800 rounded-lg transition-colors duration-200"
            aria-label="Settings"
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </div>
    </header>
  );
};

export default Header; 