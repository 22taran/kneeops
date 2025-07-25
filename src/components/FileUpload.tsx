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
        className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors duration-200 cursor-pointer ${
          isDragging 
            ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20' 
            : 'border-medical-200 dark:border-medical-700 hover:border-primary-400 dark:hover:border-primary-500 bg-white dark:bg-medical-800/50'
        }`}
      >
        <input 
          {...getInputProps()} 
          ref={fileInputRef}
          type="file"
          accept={acceptedFileTypes.join(',')}
          onChange={handleFileChange}
          className="hidden"
        />
        
        <div className="flex flex-col items-center justify-center space-y-3">
          <div className="p-3 rounded-full bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400">
            <UploadIcon className="w-6 h-6" />
          </div>
          
          <div className="space-y-1">
            <h3 className="text-lg font-medium text-medical-900 dark:text-white">
              {isDragging ? 'Drop the file here' : 'Drag and drop your MRI file'}
            </h3>
            <p className="text-sm text-medical-500 dark:text-medical-400">
              {acceptedFileTypes.map(ext => ext.toUpperCase()).join(', ')} files up to 50MB
            </p>
          </div>
          
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
            className="mt-2 px-4 py-2 text-sm font-medium text-primary-600 dark:text-primary-400 hover:text-primary-500 dark:hover:text-primary-300 transition-colors"
          >
            or select a file
          </button>
        </div>
      </div>
      
      {/* Selected File Preview */}
      {file && (
        <div className="p-4 rounded-lg bg-medical-50 dark:bg-medical-800/50 border border-medical-200 dark:border-medical-700">
        <div className="flex items-center justify-between mt-4">
          <div className="flex items-center space-x-2">
            <File className="h-4 w-4 text-gray-400" />
            <span className="text-sm font-medium text-gray-300">
              {file.name}
            </span>
            <span className="text-xs text-gray-500">
              {formatFileSize(file.size)}
            </span>
          </div>
          <div className="flex space-x-2">
            <button
              type="button"
              onClick={() => {
                setFile(null);
                if (fileInputRef.current) {
                  fileInputRef.current.value = '';
                }
              }}
              className="inline-flex items-center px-3 py-1.5 border border-gray-300 shadow-sm text-xs font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
            >
              <X className="h-3 w-3 mr-1" />
              Remove
            </button>
          </div>
        </div>
      </div>
      )}
      
      {/* Error Message */}
      {error && (
        <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 flex items-start space-x-2">
          <AlertCircle className="w-5 h-5 text-red-500 dark:text-red-400 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
        </div>
      )}
      
      {/* Upload Button */}
      <div className="flex flex-col sm:flex-row gap-2 w-full">
        <button
          type="button"
          onClick={handleUpload}
          disabled={!file || isUploading}
          className={
            'w-full py-2.5 px-4 rounded-lg text-white font-medium flex items-center justify-center space-x-2 transition-all duration-200 ' +
            (!file || isUploading
              ? 'bg-medical-300 dark:bg-medical-700 cursor-not-allowed'
              : 'bg-primary-600 hover:bg-primary-700 dark:bg-primary-600 dark:hover:bg-primary-700 shadow-sm hover:shadow-md')
          }
        >
          {isUploading ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Uploading...
            </>
          ) : (
            'Upload File'
          )}
        </button>
      </div>
    </div>
  );
};

export default FileUpload;