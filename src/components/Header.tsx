import React from 'react';
import { Activity, Settings, Bell, HelpCircle } from 'lucide-react';
import ThemeToggle from './ThemeToggle';

interface HeaderProps {
  className?: string;
}

const Header: React.FC<HeaderProps> = ({ className = '' }) => {
  return (
    <header className={`bg-white/95 dark:bg-medical-900/95 backdrop-blur-md border-b border-medical-200/50 dark:border-medical-800/50 px-4 sm:px-6 lg:px-8 py-4 shadow-sm sticky top-0 z-50 ${className}`}>
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="relative">
            <div className="w-12 h-12 bg-gradient-to-br from-primary-500 to-primary-700 rounded-xl flex items-center justify-center shadow-lg shadow-primary-500/20 dark:shadow-primary-900/30">
              <Activity className="w-7 h-7 text-white" />
            </div>
            <div className="absolute -top-1 -right-1 w-4 h-4 bg-success-500 rounded-full border-2 border-white dark:border-medical-900 animate-pulse"></div>
          </div>
          <div className="hidden sm:block">
            <h1 className="text-2xl font-bold bg-gradient-to-r from-primary-600 to-primary-800 dark:from-primary-400 dark:to-primary-600 bg-clip-text text-transparent">
              KneeOps
            </h1>
            <p className="text-xs text-medical-600 dark:text-medical-400 font-medium">AI-Powered Knee MRI Analysis</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-1 sm:space-x-2">
          <button 
            className="p-2.5 text-medical-600 dark:text-medical-300 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-primary-50 dark:hover:bg-medical-800 rounded-xl transition-all duration-200 hover:scale-105 active:scale-95"
            aria-label="Help"
          >
            <HelpCircle className="w-5 h-5" />
          </button>
          <button 
            className="p-2.5 text-medical-600 dark:text-medical-300 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-primary-50 dark:hover:bg-medical-800 rounded-xl transition-all duration-200 hover:scale-105 active:scale-95 relative"
            aria-label="Notifications"
          >
            <Bell className="w-5 h-5" />
            <span className="absolute top-1.5 right-1.5 w-2 h-2 bg-error-500 rounded-full ring-2 ring-white dark:ring-medical-900 animate-pulse"></span>
          </button>
          <ThemeToggle />
          <div className="h-8 w-px bg-medical-200 dark:bg-medical-700 mx-2"></div>
          <button 
            className="p-2.5 text-medical-600 dark:text-medical-300 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-primary-50 dark:hover:bg-medical-800 rounded-xl transition-all duration-200 hover:scale-105 active:scale-95"
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