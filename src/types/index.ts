export interface ChatMessage {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
  isLoading?: boolean;
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
  acceptedFileTypes: string[];
} 