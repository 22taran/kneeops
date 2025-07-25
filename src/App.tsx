import React, { useState, useCallback, useRef, useEffect } from 'react';
import { ThemeProvider } from './context/ThemeContext';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import ChatInterface from './components/ChatInterface';
import { ChatMessage } from './types';
import apiService, { ChatResponse } from './services/api';
import { cn } from './utils';

const App: React.FC = () => {
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [setMriData] = useState<any>(null);
  const [isAppLoading, setIsAppLoading] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Initialize with a welcome message
  useEffect(() => {
    if (chatMessages.length === 0) {
      setChatMessages([{
        id: 'welcome',
        content: 'Welcome to KneeOps! Please upload an MRI scan to begin analysis.',
        role: 'assistant',
        timestamp: new Date(),
      }]);
    }
  }, [chatMessages.length]);

  // Simulate initial loading
  useEffect(() => {
    const timer = setTimeout(() => {
      setIsAppLoading(false);
    }, 1000);
    return () => clearTimeout(timer);
  }, []);

  const handleFileUpload = useCallback(async (file: File) => {
    setIsUploading(true);
    setIsProcessing(true);
    
    try {
      console.log('Starting file upload in App.tsx');
      
      // Store file for Grad-CAM visualization
      if (fileInputRef.current) {
        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(file);
        fileInputRef.current.files = dataTransfer.files;
      }
      
      // Upload file to ML model
      const mlResponse = await apiService.uploadMRI(file);
      
      console.log('Received ML response:', mlResponse);
      
      if (!mlResponse.success) {
        throw new Error(mlResponse.message || 'Failed to upload file');
      }
      
      // Update state with MRI data and predictions
      if (mlResponse.mri_data) {
        // Ensure the MRI data is in the correct format for the viewer
        const formattedMriData = Array.isArray(mlResponse.mri_data) 
          ? mlResponse.mri_data 
          : [mlResponse.mri_data];
        setMriData(formattedMriData);
        console.log('Set MRI data:', formattedMriData);
      }
      

      
      // Create analysis results message
      const analysisMessage: ChatMessage = {
        id: `analysis-${Date.now()}`,
        content: `I've analyzed the MRI file "${file.name}". Here are the results:`, 
        role: 'assistant',
        timestamp: new Date(),
      };
      
      // Create structured analysis results
      if (mlResponse.prediction) {
        // Single image analysis
        const prediction = mlResponse.prediction;
        const analysisResults = {
          averageConfidence: (prediction.confidence_percentage || 0) / 100,
          diagnosis: prediction.class || 'No diagnosis',
          diagnosisConfidence: (prediction.confidence_percentage || 0) / 100,
          processedImages: 1
        };
        
        // Add analysis results to the message
        analysisMessage.analysisResults = analysisResults;
        
        // Add recommendations as a follow-up message if available
        if (prediction.recommendations && prediction.recommendations.length > 0) {
          const recommendationsMessage: ChatMessage = {
            id: `recs-${Date.now()}`,
            content: `**Recommendations:**\n${prediction.recommendations.map((r: string) => `• ${r}`).join('\n')}`,
            role: 'assistant',
            timestamp: new Date(),
          };
          setChatMessages(prev => [...prev, analysisMessage, recommendationsMessage]);
        } else {
          setChatMessages(prev => [...prev, analysisMessage]);
        }
      } else if (mlResponse.predictions && mlResponse.predictions.length > 0) {
        // Batch analysis - calculate average confidence
        const totalConfidence = mlResponse.predictions.reduce(
          (sum: number, p: any) => sum + (p.confidence_percentage || 0), 0
        );
        const avgConfidence = totalConfidence / mlResponse.predictions.length;
        
        // Find most common diagnosis
        const diagnosisCounts = (mlResponse.predictions as Array<{class?: string}>).reduce(
          (counts: Record<string, number>, p) => {
            const className = p.class || 'Unknown';
            counts[className] = (counts[className] || 0) + 1;
            return counts;
          }, 
          {} as Record<string, number>
        );
        
        // Find the most common diagnosis
        type DiagnosisCount = [string, number];
        const mostCommonDiagnosis = (Object.entries(diagnosisCounts) as DiagnosisCount[]).reduce(
          (a: DiagnosisCount, b: DiagnosisCount) => (a[1] > b[1] ? a : b),
          ['Unknown', 0] as DiagnosisCount
        );
        
        const analysisResults = {
          averageConfidence: avgConfidence / 100,
          diagnosis: mostCommonDiagnosis[0],
          diagnosisConfidence: (mostCommonDiagnosis[1] / mlResponse.predictions.length) * (avgConfidence / 100),
          processedImages: mlResponse.predictions.length
        };
        
        // Add analysis results to the message
        analysisMessage.analysisResults = analysisResults;
        setChatMessages(prev => [...prev, analysisMessage]);
      } else {
        // Fallback if no predictions are available
        analysisMessage.content = "I couldn't analyze the MRI file. Please try again or upload a different file.";
        setChatMessages(prev => [...prev, analysisMessage]);
      }

    } catch (error) {
      console.error('Error uploading file:', error);
      
      // Add error message
      const errorMessage: ChatMessage = {
        id: Date.now().toString(),
        content: `Sorry, I encountered an error while processing the MRI file. Please try again or contact support if the problem persists.`,
        role: 'assistant',
        timestamp: new Date(),
      };
      
      setChatMessages([errorMessage]);
    } finally {
      setIsUploading(false);
      setIsProcessing(false);
    }
  }, []);

  const handleSendMessage = useCallback(async (message: string) => {
    // Add user message
    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      content: message,
      role: 'user',
      timestamp: new Date(),
    };
    
    setChatMessages(prev => [...prev, userMessage]);
    
    // Add loading message
    const loadingMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      content: '',
      role: 'assistant',
      timestamp: new Date(),
      isLoading: true,
    };
    
    setChatMessages(prev => [...prev, loadingMessage]);
    
    try {
      // Send message to ML model
      const chatResponse: ChatResponse = await apiService.sendChatMessage(message, 'file-id');
      
      if (!chatResponse.success) {
        throw new Error('Failed to get response from AI');
      }
      
      // Replace loading message with actual response
      setChatMessages(prev => 
        prev.map(msg => 
          msg.id === loadingMessage.id 
            ? { ...msg, content: chatResponse.response, isLoading: false }
            : msg
        )
      );
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Replace loading message with error
      setChatMessages(prev => 
        prev.map(msg => 
          msg.id === loadingMessage.id 
            ? { ...msg, content: 'Sorry, I encountered an error while processing your question. Please try again.', isLoading: false }
            : msg
        )
      );
    }
  }, []);

  // Loading state
  if (isAppLoading) {
    return (
      <div className="min-h-screen bg-white dark:bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Loading KneeOps</h2>
          <p className="text-gray-600 dark:text-gray-400">Preparing your workspace...</p>
        </div>
      </div>
    );
  }

  return (
    <ThemeProvider>
      <div className="flex flex-col min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors duration-200">
        <Header className="flex-shrink-0" />
        
        <main className="flex-1 container mx-auto px-4 py-4 min-h-0">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 h-full min-h-[calc(100vh-200px)]">
            
            {/* MRI Viewer */}
            <div className="lg:col-span-1 flex flex-col space-y-4 max-h-full">
              <FileUpload 
                onFileUpload={handleFileUpload}
                isUploading={isUploading}
                fileInputRef={fileInputRef}
              />
              
            </div>
            
            {/* Right Column - Chat Interface */}
            <div className="lg:col-span-2 bg-white dark:bg-gray-800 rounded-xl shadow-sm border border-gray-200 dark:border-gray-700 flex flex-col min-h-0">
              <div className={cn("flex-1 flex flex-col min-h-0", {
                "opacity-50 pointer-events-none": isProcessing
              })}>
                <ChatInterface 
                  messages={chatMessages} 
                  onSendMessage={handleSendMessage} 
                  disabled={isProcessing || isUploading}
                  placeholder={
                    chatMessages.length === 0 
                      ? "Upload an MRI file to begin analysis..." 
                      : "Ask me anything about this MRI scan..."
                  }
                  className="h-full"
                />
              </div>
            </div>
          </div>
        </main>
        
        <footer className="flex-shrink-0 py-4 px-6 border-t border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900 text-center text-sm text-gray-500 dark:text-gray-400">
          <p>KneeOps AI - Not for diagnostic use. Always consult with a healthcare professional.</p>
        </footer>
      </div>
    </ThemeProvider>
  );
};

export default App;