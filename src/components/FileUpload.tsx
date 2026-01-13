import React, { useState, useCallback, useRef } from 'react';
import { Upload as UploadIcon, File, X, Loader2, AlertCircle } from 'lucide-react';
import { useDropzone } from 'react-dropzone';

interface FileUploadProps {
  onFileUpload: (file: File) => Promise<void>;
  isUploading: boolean;
  acceptedFileTypes?: string[];
  fileInputRef?: React.RefObject<HTMLInputElement>;
}

const FileUpload: React.FC<FileUploadProps> = ({ 
  onFileUpload, 
  isUploading, 
  acceptedFileTypes = ['.pck'],
  fileInputRef: externalFileInputRef 
}) => {
  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const internalFileInputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = externalFileInputRef || internalFileInputRef;

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const selectedFile = acceptedFiles[0];
    if (selectedFile && acceptedFileTypes.some(ext => selectedFile.name.toLowerCase().endsWith(ext.toLowerCase()))) {
      setFile(selectedFile);
      setError(null);
    } else {
      setError(`Please select a valid ${acceptedFileTypes.join(' or ')} file`);
      setFile(null);
    }
    setIsDragging(false);
  }, [acceptedFileTypes]);

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      'application/octet-stream': ['.pck'],
      'application/x-python-pickle': ['.pck']
    },
    multiple: false,
    onDragEnter: () => setIsDragging(true),
    onDragLeave: () => setIsDragging(false),
  });

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile && acceptedFileTypes.some(ext => selectedFile.name.toLowerCase().endsWith(ext.toLowerCase()))) {
      setFile(selectedFile);
      setError(null);
    } else {
      setError(`Please select a valid ${acceptedFileTypes.join(' or ')} file`);
      setFile(null);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    
    try {
      setError(null);
      await onFileUpload(file);
    } catch (err) {
      console.error('Upload failed:', err);
      setError('Upload failed. Please try again.');
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="space-y-6">
      {/* Drag and Drop Zone */}
      <div 
        {...getRootProps()} 
        className={`border-2 border-dashed rounded-2xl p-10 text-center transition-all duration-300 cursor-pointer relative overflow-hidden group ${
          isDragging 
            ? 'border-primary-500 bg-gradient-to-br from-primary-50 to-primary-100/50 dark:from-primary-900/30 dark:to-primary-800/20 scale-[1.02] shadow-lg shadow-primary-500/10' 
            : 'border-medical-300 dark:border-medical-600 hover:border-primary-400 dark:hover:border-primary-500 bg-gradient-to-br from-white to-medical-50/50 dark:from-medical-800/30 dark:to-medical-900/50 hover:shadow-xl hover:shadow-primary-500/5'
        }`}
      >
        <div className="absolute inset-0 bg-gradient-to-br from-primary-500/0 to-primary-600/0 group-hover:from-primary-500/5 group-hover:to-primary-600/5 transition-all duration-300"></div>
        <input 
          {...getInputProps()} 
          ref={fileInputRef}
          type="file"
          accept={acceptedFileTypes.join(',')}
          onChange={handleFileChange}
          className="hidden"
        />
        
        <div className="flex flex-col items-center justify-center space-y-4 relative z-10">
          <div className={`p-4 rounded-2xl transition-all duration-300 ${
            isDragging 
              ? 'bg-primary-500 text-white scale-110 shadow-lg shadow-primary-500/30' 
              : 'bg-gradient-to-br from-primary-100 to-primary-200 dark:from-primary-900/40 dark:to-primary-800/40 text-primary-600 dark:text-primary-400 group-hover:scale-105'
          }`}>
            <UploadIcon className="w-8 h-8" />
          </div>
          
          <div className="space-y-2">
            <h3 className="text-xl font-semibold text-medical-900 dark:text-white">
              {isDragging ? 'Drop the file here' : 'Drag and drop your MRI file'}
            </h3>
            <p className="text-sm text-medical-600 dark:text-medical-400">
              {acceptedFileTypes.map(ext => ext.toUpperCase()).join(', ')} files up to 50MB
            </p>
          </div>
          
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
            className="mt-2 px-6 py-2.5 text-sm font-semibold text-primary-600 dark:text-primary-400 hover:text-primary-700 dark:hover:text-primary-300 bg-primary-50 dark:bg-primary-900/30 hover:bg-primary-100 dark:hover:bg-primary-900/50 rounded-lg transition-all duration-200 hover:scale-105 active:scale-95"
          >
            or select a file
          </button>
        </div>
      </div>
      
      {/* Selected File Preview */}
      {file && (
        <div className="p-5 rounded-xl bg-gradient-to-br from-medical-50 to-white dark:from-medical-800/50 dark:to-medical-900/30 border border-medical-200 dark:border-medical-700 shadow-sm animate-slide-up">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3 flex-1 min-w-0">
              <div className="flex-shrink-0 p-2 rounded-lg bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400">
                <File className="h-5 w-5" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-semibold text-medical-900 dark:text-white truncate">
                  {file.name}
                </p>
                <p className="text-xs text-medical-500 dark:text-medical-400">
                  {formatFileSize(file.size)}
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={() => {
                setFile(null);
                if (fileInputRef.current) {
                  fileInputRef.current.value = '';
                }
              }}
              className="ml-3 inline-flex items-center px-3 py-2 border border-medical-300 dark:border-medical-600 shadow-sm text-xs font-medium rounded-lg text-medical-700 dark:text-medical-300 bg-white dark:bg-medical-800 hover:bg-medical-50 dark:hover:bg-medical-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-all duration-200 hover:scale-105 active:scale-95"
            >
              <X className="h-3.5 w-3.5 mr-1.5" />
              Remove
            </button>
          </div>
        </div>
      )}
      
      {/* Error Message */}
      {error && (
        <div className="p-4 rounded-xl bg-gradient-to-br from-error-50 to-error-100/50 dark:from-error-900/30 dark:to-error-800/20 border border-error-200 dark:border-error-800 flex items-start space-x-3 animate-slide-up shadow-sm">
          <AlertCircle className="w-5 h-5 text-error-500 dark:text-error-400 flex-shrink-0 mt-0.5" />
          <p className="text-sm font-medium text-error-700 dark:text-error-300">{error}</p>
        </div>
      )}
      
      {/* Upload Button */}
      <div className="flex flex-col sm:flex-row gap-3 w-full">
        <button
          type="button"
          onClick={handleUpload}
          disabled={!file || isUploading}
          className={
            'w-full py-3.5 px-6 rounded-xl text-white font-semibold flex items-center justify-center space-x-2 transition-all duration-200 shadow-lg ' +
            (!file || isUploading
              ? 'bg-medical-300 dark:bg-medical-700 cursor-not-allowed shadow-none'
              : 'bg-gradient-to-r from-primary-600 to-primary-700 hover:from-primary-700 hover:to-primary-800 dark:from-primary-600 dark:to-primary-700 dark:hover:from-primary-700 dark:hover:to-primary-800 hover:shadow-xl hover:shadow-primary-500/30 hover:scale-[1.02] active:scale-[0.98]')
          }
        >
          {isUploading ? (
            <>
              <Loader2 className="h-5 w-5 animate-spin" />
              <span>Uploading...</span>
            </>
          ) : (
            <>
              <UploadIcon className="h-5 w-5" />
              <span>Upload File</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default FileUpload;