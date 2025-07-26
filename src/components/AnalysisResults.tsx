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
    <div className="bg-white dark:bg-medical-800 rounded-lg border border-medical-200 dark:border-medical-700 overflow-hidden">
      <div className="bg-gradient-to-r from-primary-50 to-medical-50 dark:from-medical-900 dark:to-medical-800 p-4 border-b border-medical-200 dark:border-medical-700">
        <h3 className="text-lg font-semibold text-medical-900 dark:text-white">Analysis Complete</h3>
        <p className="text-sm text-medical-500 dark:text-medical-400">
          Processed {processedImages} image{processedImages !== 1 ? 's' : ''} with {displayConfidence}% confidence
        </p>
      </div>
      
      <div className="p-4 space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-medical-50 dark:bg-medical-900/50 p-4 rounded-lg">
            <h4 className="text-sm font-medium text-medical-500 dark:text-medical-400 mb-2">Diagnosis</h4>
            <p className={`text-xl font-semibold ${getSeverityClass()}`}>
              {diagnosis || 'No diagnosis available'}
            </p>
          </div>
          <div className="bg-medical-50 dark:bg-medical-900/50 p-4 rounded-lg">
            <h4 className="text-sm font-medium text-medical-500 dark:text-medical-400 mb-2">Confidence</h4>
            <p className="text-xl font-semibold text-primary-600 dark:text-primary-400">
              {displayConfidence}%
              {diagnosisConfidence < 0.7 && (
                <span className="text-xs text-yellow-600 dark:text-yellow-400 block mt-1">
                  Low confidence - Please consult a specialist
                </span>
              )}
            </p>
          </div>
        </div>

        <div>
          <h4 className="text-sm font-medium text-medical-500 dark:text-medical-400 mb-3">Recommendations</h4>
          <ul className="space-y-2">
            <li className="flex items-start">
              <CheckCircle2 className="h-5 w-5 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
              <span>Analyzed {processedImages} MRI image{processedImages !== 1 ? 's' : ''}</span>
            </li>
            <li className="flex items-start">
              <CheckCircle2 className="h-5 w-5 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
              <span>Average confidence: {(averageConfidence * 100).toFixed(1)}%</span>
            </li>
            <li className="flex items-start">
              <CheckCircle2 className="h-5 w-5 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
              <span>✅ No significant injuries detected</span>
            </li>
            <li className="flex items-start">
              <CheckCircle2 className="h-5 w-5 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
              <span>Continue with normal activities</span>
            </li>
            <li className="flex items-start">
              <AlertCircle className="h-5 w-5 text-yellow-500 mr-2 mt-0.5 flex-shrink-0" />
              <span>This analysis is for screening purposes only</span>
            </li>
            <li className="flex items-start">
              <AlertCircle className="h-5 w-5 text-yellow-500 mr-2 mt-0.5 flex-shrink-0" />
              <span>Final diagnosis should be made by healthcare professional</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default AnalysisResults;
