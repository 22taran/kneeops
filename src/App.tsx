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
      const response = await apiService.uploadMRI(file);
      
      console.log('Received ML response:', response);
      
      if (!response.success) {
        throw new Error(response.message || 'Failed to upload file');
      }
      
      // Extract the data from the response
      const mlResponse = response;
      
      // Update state with MRI data and predictions
      if (mlResponse.mri_data) {
        // Ensure the MRI data is in the correct format for the viewer
        const formattedMriData = Array.isArray(mlResponse.mri_data) 
          ? mlResponse.mri_data 
          : [mlResponse.mri_data];
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
        // Convert confidence to a number and ensure it's between 0 and 1
        const confidenceValue = parseFloat(prediction.confidence || prediction.confidence_percentage || '0');
        const normalizedConfidence = confidenceValue > 1 ? confidenceValue / 100 : confidenceValue;
        
        const analysisResults = {
          averageConfidence: Math.min(Math.max(normalizedConfidence, 0), 1), // Ensure between 0 and 1
          diagnosis: prediction.class || prediction.class_name || 'No diagnosis',
          diagnosisConfidence: normalizedConfidence,
          processedImages: mlResponse.total_images || 1
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
        } else if (mlResponse.overall_analysis?.overall_recommendations) {
          // Use overall recommendations if available
          const recommendationsMessage: ChatMessage = {
            id: `recs-${Date.now()}`,
            content: `**Recommendations:**\n${mlResponse.overall_analysis.overall_recommendations.join('\n• ')}`,
            role: 'assistant',
            timestamp: new Date(),
          };
          setChatMessages(prev => [...prev, analysisMessage, recommendationsMessage]);
        } else {
          setChatMessages(prev => [...prev, analysisMessage]);
        }
      } else if (mlResponse.predictions && mlResponse.predictions.length > 0) {
        // Batch analysis - process all predictions
        const predictions = Array.isArray(mlResponse.predictions) ? mlResponse.predictions : [];
        
        // Process all predictions to normalize confidence scores
        const processedPredictions = predictions.map((p: any) => {
          const confidence = parseFloat(p.confidence || p.confidence_percentage || '0');
          return {
            ...p,
            confidence: confidence > 1 ? confidence / 100 : confidence, // Normalize to 0-1 range
            className: p.class || p.prediction?.class || 'Unknown'
          };
        });
        
        // Calculate average confidence from normalized values
        const totalConfidence = processedPredictions.reduce(
          (sum: number, p: any) => sum + p.confidence, 0
        );
        const avgConfidence = processedPredictions.length > 0 
          ? totalConfidence / processedPredictions.length 
          : 0;
        
        // Find most common diagnosis
        const diagnosisCounts = processedPredictions.reduce(
          (counts: Record<string, number>, p: any) => {
            counts[p.className] = (counts[p.className] || 0) + 1;
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
          averageConfidence: Math.min(Math.max(avgConfidence, 0), 1), // Ensure between 0 and 1
          diagnosis: mostCommonDiagnosis[0],
          diagnosisConfidence: avgConfidence,
          processedImages: processedPredictions.length
        };
        
        // Add analysis results to the message
        analysisMessage.analysisResults = analysisResults;
        
        // Add recommendations from the first prediction if available
        if (predictions[0]?.recommendations?.length > 0) {
          const recommendationsMessage: ChatMessage = {
            id: `recs-${Date.now()}`,
            content: `**Recommendations:**\n${predictions[0].recommendations.map((r: string) => `• ${r}`).join('\n')}`,
            role: 'assistant',
            timestamp: new Date(),
          };
          setChatMessages(prev => [...prev, analysisMessage, recommendationsMessage]);
        } else {
          setChatMessages(prev => [...prev, analysisMessage]);
        }
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
      <div className="min-h-screen bg-gradient-to-br from-primary-50 via-white to-medical-50 dark:from-medical-900 dark:via-medical-800 dark:to-medical-900 flex items-center justify-center">
        <div className="text-center animate-fade-in">
          <div className="relative">
            <div className="w-20 h-20 border-4 border-primary-200 dark:border-primary-800 rounded-full mx-auto mb-2"></div>
            <div className="w-20 h-20 border-4 border-primary-600 border-t-transparent rounded-full animate-spin mx-auto absolute top-0 left-1/2 transform -translate-x-1/2"></div>
          </div>
          <h2 className="text-2xl font-bold bg-gradient-to-r from-primary-600 to-primary-800 dark:from-primary-400 dark:to-primary-600 bg-clip-text text-transparent mt-6 mb-2">
            Loading KneeOps
          </h2>
          <p className="text-medical-600 dark:text-medical-400">Preparing your workspace...</p>
        </div>
      </div>
    );
  }

  return (
    <ThemeProvider>
      <div className="flex flex-col h-screen max-h-screen bg-gradient-to-br from-medical-50 via-white to-primary-50/30 dark:from-medical-950 dark:via-medical-900 dark:to-medical-950 transition-colors duration-300 overflow-hidden">
        <Header className="flex-shrink-0" />
        
        <main className="flex-1 container mx-auto px-4 sm:px-6 lg:px-8 py-6 lg:py-8 min-h-0 flex flex-col overflow-hidden">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 lg:gap-8 flex-1 min-h-0 max-h-full">
            
            {/* MRI Viewer */}
            <div className="lg:col-span-1 flex flex-col space-y-6 min-h-0">
              <div className="bg-white dark:bg-medical-800 rounded-2xl shadow-lg border border-medical-200 dark:border-medical-700 p-6 backdrop-blur-sm">
                <FileUpload 
                  onFileUpload={handleFileUpload}
                  isUploading={isUploading}
                  fileInputRef={fileInputRef}
                />
              </div>
            </div>
            
            {/* Right Column - Chat Interface */}
            <div className="lg:col-span-2 bg-white dark:bg-medical-800 rounded-2xl shadow-lg border border-medical-200 dark:border-medical-700 flex flex-col h-full max-h-full min-h-0 overflow-hidden backdrop-blur-sm">
              <div className={cn("flex flex-col h-full max-h-full min-h-0 overflow-hidden", {
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
                  className="h-full max-h-full"
                />
              </div>
            </div>
          </div>
        </main>
        
        <footer className="flex-shrink-0 py-6 px-6 border-t border-medical-200 dark:border-medical-800 bg-white/80 dark:bg-medical-900/80 backdrop-blur-sm text-center">
          <p className="text-sm text-medical-600 dark:text-medical-400">
            <span className="font-semibold text-primary-600 dark:text-primary-400">KneeOps AI</span>
            {' '}— Not for diagnostic use. Always consult with a healthcare professional.
          </p>
        </footer>
      </div>
    </ThemeProvider>
  );
};

export default App;