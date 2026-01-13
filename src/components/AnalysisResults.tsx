import React from 'react';
import { CheckCircle2, AlertCircle } from 'lucide-react';

export interface AnalysisResultsProps {
  averageConfidence: number;
  diagnosis: string;
  diagnosisConfidence: number;
  processedImages: number;
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({
  averageConfidence,
  diagnosis,
  diagnosisConfidence,
  processedImages,
}) => {
  // Format confidence for display
  const displayConfidence = typeof diagnosisConfidence === 'number' 
    ? (diagnosisConfidence * 100).toFixed(1)
    : (averageConfidence * 100).toFixed(1);
    
  // Get severity class for styling
  const getSeverityClass = () => {
    if (diagnosis?.toLowerCase().includes('healthy')) return 'text-green-600 dark:text-green-400';
    if (diagnosis?.toLowerCase().includes('acl')) return 'text-yellow-600 dark:text-yellow-400';
    if (diagnosis?.toLowerCase().includes('meniscus')) return 'text-red-600 dark:text-red-400';
    return 'text-medical-900 dark:text-white';
  };

  return (
    <div className="bg-white dark:bg-medical-800 rounded-xl border border-medical-200 dark:border-medical-700 overflow-hidden shadow-lg">
      <div className="bg-gradient-to-r from-primary-500 via-primary-600 to-primary-700 dark:from-primary-600 dark:via-primary-700 dark:to-primary-800 p-5 border-b border-primary-600/20">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xl font-bold text-white mb-1">Analysis Complete</h3>
            <p className="text-sm text-primary-100">
              Processed {processedImages} image{processedImages !== 1 ? 's' : ''} with {displayConfidence}% confidence
            </p>
          </div>
          <div className="w-12 h-12 bg-white/20 rounded-xl flex items-center justify-center backdrop-blur-sm">
            <CheckCircle2 className="w-7 h-7 text-white" />
          </div>
        </div>
      </div>
      
      <div className="p-6 space-y-6 bg-gradient-to-br from-white to-medical-50/50 dark:from-medical-800 dark:to-medical-900/50">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="bg-gradient-to-br from-medical-50 to-white dark:from-medical-900/50 dark:to-medical-800/50 p-5 rounded-xl border border-medical-200 dark:border-medical-700 shadow-sm hover:shadow-md transition-shadow duration-200">
            <h4 className="text-xs font-semibold text-medical-500 dark:text-medical-400 mb-3 uppercase tracking-wide">Diagnosis</h4>
            <p className={`text-2xl font-bold ${getSeverityClass()}`}>
              {diagnosis || 'No diagnosis available'}
            </p>
          </div>
          <div className="bg-gradient-to-br from-primary-50 to-primary-100/50 dark:from-primary-900/30 dark:to-primary-800/20 p-5 rounded-xl border border-primary-200 dark:border-primary-800 shadow-sm hover:shadow-md transition-shadow duration-200">
            <h4 className="text-xs font-semibold text-primary-600 dark:text-primary-400 mb-3 uppercase tracking-wide">Confidence</h4>
            <div className="flex items-baseline space-x-2">
              <p className="text-2xl font-bold text-primary-600 dark:text-primary-400">
                {displayConfidence}%
              </p>
              {diagnosisConfidence < 0.7 && (
                <span className="text-xs font-medium text-warning-600 dark:text-warning-400 bg-warning-50 dark:bg-warning-900/30 px-2 py-1 rounded-md">
                  Low confidence
                </span>
              )}
            </div>
            {diagnosisConfidence < 0.7 && (
              <p className="text-xs text-warning-700 dark:text-warning-300 mt-2">
                Please consult a specialist
              </p>
            )}
          </div>
        </div>

        <div className="bg-white dark:bg-medical-800/50 p-5 rounded-xl border border-medical-200 dark:border-medical-700">
          <h4 className="text-sm font-semibold text-medical-700 dark:text-medical-300 mb-4 flex items-center">
            <span className="w-1 h-5 bg-primary-500 rounded-full mr-2"></span>
            Recommendations
          </h4>
          <ul className="space-y-3">
            <li className="flex items-start group">
              <div className="flex-shrink-0 w-6 h-6 bg-success-100 dark:bg-success-900/30 rounded-lg flex items-center justify-center mr-3 mt-0.5 group-hover:scale-110 transition-transform duration-200">
                <CheckCircle2 className="h-4 w-4 text-success-600 dark:text-success-400" />
              </div>
              <span className="text-sm text-medical-700 dark:text-medical-300 pt-0.5">Analyzed {processedImages} MRI image{processedImages !== 1 ? 's' : ''}</span>
            </li>
            <li className="flex items-start group">
              <div className="flex-shrink-0 w-6 h-6 bg-success-100 dark:bg-success-900/30 rounded-lg flex items-center justify-center mr-3 mt-0.5 group-hover:scale-110 transition-transform duration-200">
                <CheckCircle2 className="h-4 w-4 text-success-600 dark:text-success-400" />
              </div>
              <span className="text-sm text-medical-700 dark:text-medical-300 pt-0.5">Average confidence: {(averageConfidence * 100).toFixed(1)}%</span>
            </li>
            <li className="flex items-start group">
              <div className="flex-shrink-0 w-6 h-6 bg-success-100 dark:bg-success-900/30 rounded-lg flex items-center justify-center mr-3 mt-0.5 group-hover:scale-110 transition-transform duration-200">
                <CheckCircle2 className="h-4 w-4 text-success-600 dark:text-success-400" />
              </div>
              <span className="text-sm text-medical-700 dark:text-medical-300 pt-0.5">No significant injuries detected</span>
            </li>
            <li className="flex items-start group">
              <div className="flex-shrink-0 w-6 h-6 bg-success-100 dark:bg-success-900/30 rounded-lg flex items-center justify-center mr-3 mt-0.5 group-hover:scale-110 transition-transform duration-200">
                <CheckCircle2 className="h-4 w-4 text-success-600 dark:text-success-400" />
              </div>
              <span className="text-sm text-medical-700 dark:text-medical-300 pt-0.5">Continue with normal activities</span>
            </li>
            <li className="flex items-start group">
              <div className="flex-shrink-0 w-6 h-6 bg-warning-100 dark:bg-warning-900/30 rounded-lg flex items-center justify-center mr-3 mt-0.5 group-hover:scale-110 transition-transform duration-200">
                <AlertCircle className="h-4 w-4 text-warning-600 dark:text-warning-400" />
              </div>
              <span className="text-sm text-medical-700 dark:text-medical-300 pt-0.5">This analysis is for screening purposes only</span>
            </li>
            <li className="flex items-start group">
              <div className="flex-shrink-0 w-6 h-6 bg-warning-100 dark:bg-warning-900/30 rounded-lg flex items-center justify-center mr-3 mt-0.5 group-hover:scale-110 transition-transform duration-200">
                <AlertCircle className="h-4 w-4 text-warning-600 dark:text-warning-400" />
              </div>
              <span className="text-sm text-medical-700 dark:text-medical-300 pt-0.5">Final diagnosis should be made by healthcare professional</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default AnalysisResults;
