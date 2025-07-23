import React, { useState } from 'react';
import apiService from '../services/api';

interface FileUploadProps {
  onFileUpload: (file: File) => Promise<void>;
  isUploading: boolean;
  acceptedFileTypes: string[];
}

interface DetailedAnalysisResult {
  success: boolean;
  prediction?: any;
  predictions?: any[];
  total_images: number;
  overall_analysis: {
    total_images: number;
    processed_images: number;
    average_confidence: number;
    most_common_prediction: string;
    prediction_distribution: Record<string, number>;
    severity_analysis: {
      healthy_count: number;
      injury_count: number;
      severe_injury_count: number;
    };
    overall_severity: string;
    overall_recommendations: string[];
  };
}

const FileUpload: React.FC<FileUploadProps> = ({ onFileUpload, isUploading, acceptedFileTypes }) => {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<DetailedAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile && selectedFile.name.endsWith('.pck')) {
      setFile(selectedFile);
      setError(null);
      setResult(null);
    } else {
      setError('Please select a valid .pck file');
      setFile(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    setError(null);

    try {
      // Call the parent's onFileUpload function
      await onFileUpload(file);
      
      // Also get detailed results for display
      const response = await apiService.uploadMRI(file);
      setResult(response as any);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'healthy': return 'text-green-600';
      case 'mild_injury': return 'text-yellow-600';
      case 'severe_injury': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600';
    if (confidence >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold mb-6 text-gray-800">MRI Analysis</h2>
        
        {/* File Upload Section */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Upload MRI File (.pck)
          </label>
          <input
            type="file"
            accept=".pck"
            onChange={handleFileChange}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
          {file && (
            <p className="mt-2 text-sm text-gray-600">
              Selected: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
            </p>
          )}
        </div>

        {/* Upload Button */}
        <button
          onClick={handleUpload}
          disabled={!file || isUploading}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
        >
          {isUploading ? 'Analyzing...' : 'Analyze MRI'}
        </button>

        {/* Error Display */}
        {error && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-md">
            <p className="text-red-800">{error}</p>
          </div>
        )}

        {/* Overall Analysis Only */}
        {result && (
          <div className="mt-6">
            <div className="bg-gray-50 rounded-lg p-4">
              <h3 className="text-lg font-semibold mb-3">Overall Analysis</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="text-center">
                  <p className="text-2xl font-bold text-blue-600">{result.total_images}</p>
                  <p className="text-sm text-gray-600">Total Images</p>
                </div>
                <div className="text-center">
                  <p className={`text-2xl font-bold ${getConfidenceColor(result.overall_analysis.average_confidence)}`}>
                    {(result.overall_analysis.average_confidence * 100).toFixed(1)}%
                  </p>
                  <p className="text-sm text-gray-600">Average Confidence</p>
                </div>
                <div className="text-center">
                  <p className={`text-2xl font-bold ${getSeverityColor(result.overall_analysis.overall_severity)}`}>
                    {result.overall_analysis.overall_severity.replace('_', ' ').toUpperCase()}
                  </p>
                  <p className="text-sm text-gray-600">Overall Status</p>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUpload; 