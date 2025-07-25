export interface AnalysisResults {
  averageConfidence: number;
  diagnosis: string;
  diagnosisConfidence: number;
  processedImages: number;
}

export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  isLoading?: boolean;
  analysisResults?: AnalysisResults;
}

export interface UploadedFile {
  id: string;
  name: string;
  size: number;
  type: string;
  uploadedAt: Date;
}

export interface AppState {
  uploadedFiles: UploadedFile[];
  chatMessages: ChatMessage[];
  isUploading: boolean;
  isProcessing: boolean;
  currentFile: UploadedFile | null;
}

export interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

export interface FileUploadProps {
  onFileUpload: (file: File) => void;
  isUploading: boolean;
  acceptedFileTypes?: string[];
  fileInputRef?: React.RefObject<HTMLInputElement>;
}

export interface MRIAnalysis {
  id: string;
  fileName: string;
  timestamp: Date;
  predictions: Array<{
    class: string;
    confidence: number;
    confidence_percentage: number;
    severity?: string;
    recommendations?: string[];
  }>;
  mriData?: any; // 3D MRI data array
  metadata?: {
    width: number;
    height: number;
    depth: number;
    spacing: [number, number, number];
  };
}

export interface MRIViewerProps {
  mriData: any; // 3D MRI data array
  predictions: Array<{
    class: string;
    confidence: number;
  }>;
  onSliceChange?: (sliceIndex: number) => void;
  className?: string;
}