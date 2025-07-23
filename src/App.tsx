import React, { useState, useCallback } from 'react';
import Header from './components/Header';
import FileUpload from './components/FileUpload';
import ChatInterface from './components/ChatInterface';
import { ChatMessage, UploadedFile } from './types';
import apiService, { ChatResponse } from './services/api';

const App: React.FC = () => {
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleFileUpload = useCallback(async (file: File) => {
    setIsUploading(true);
    
    try {
      console.log('Starting file upload in App.tsx');
      
      // Upload file to ML model
      const mlResponse = await apiService.uploadMRI(file);
      
      console.log('Received ML response:', mlResponse);
      
      if (!mlResponse.success) {
        throw new Error(mlResponse.message || 'Failed to upload file');
      }
      
      setIsUploading(false);
      setIsProcessing(false);
      
      // Create welcome message with ML analysis results
      let welcomeContent = `I've analyzed the MRI file "${file.name}".\n\n`;
      
      // Handle different response formats
      if (mlResponse.prediction) {
        // Single image analysis
        const prediction = mlResponse.prediction;
        welcomeContent += `**Analysis Results:**\n`;
        welcomeContent += `• Classification: ${prediction.class}\n`;
        welcomeContent += `• Confidence: ${prediction.confidence_percentage.toFixed(1)}%\n`;
        welcomeContent += `• Severity: ${prediction.severity}\n\n`;
        
        if (prediction.recommendations && prediction.recommendations.length > 0) {
          welcomeContent += `**Recommendations:**\n`;
          prediction.recommendations.forEach((rec: string) => {
            welcomeContent += `• ${rec}\n`;
          });
        }
      } else if (mlResponse.predictions && mlResponse.predictions.length > 0) {
        // Show only the first image analysis
        const firstPrediction = mlResponse.predictions[0];
        welcomeContent += `**Image 1 Analysis:**\n`;
        welcomeContent += `• Classification: ${firstPrediction.class}\n`;
        welcomeContent += `• Confidence: ${firstPrediction.confidence_percentage.toFixed(1)}%\n`;
        welcomeContent += `• Severity: ${firstPrediction.severity}\n\n`;
        
        if (firstPrediction.recommendations && firstPrediction.recommendations.length > 0) {
          welcomeContent += `**Recommendations:**\n`;
          firstPrediction.recommendations.forEach((rec: string) => {
            welcomeContent += `• ${rec}\n`;
          });
        }
      } else {
        welcomeContent += `Analysis completed successfully. What would you like to know about this MRI?`;
      }
      
      welcomeContent += `\n\nWhat would you like to know about this analysis?`;
      
      const welcomeMessage: ChatMessage = {
        id: Date.now().toString(),
        content: welcomeContent,
        role: 'assistant',
        timestamp: new Date(),
      };
      
      console.log('Setting chat messages with:', welcomeMessage);
      setChatMessages([welcomeMessage]);
      
    } catch (error) {
      setIsUploading(false);
      setIsProcessing(false);
      console.error('Error uploading file:', error);
      
      // Add error message
      const errorMessage: ChatMessage = {
        id: Date.now().toString(),
        content: `Sorry, I encountered an error while processing the MRI file. Please try again or contact support if the problem persists.`,
        role: 'assistant',
        timestamp: new Date(),
      };
      
      setChatMessages([errorMessage]);
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

  return (
    <div className="min-h-screen bg-medical-50 flex flex-col">
      <Header />
      
      <main className="flex-1 flex flex-col lg:flex-row">
        {/* File Upload Section */}
        <div className="lg:w-1/3 p-6 border-r border-medical-200 bg-white">
          <FileUpload
            onFileUpload={handleFileUpload}
            isUploading={isUploading}
            acceptedFileTypes={['.pck']}
          />
        </div>
        
        {/* Chat Interface */}
        <div className="lg:w-2/3 flex flex-col">
          <ChatInterface
            messages={chatMessages}
            onSendMessage={handleSendMessage}
            disabled={isProcessing}
            placeholder="Upload an MRI file to start..."
          />
        </div>
      </main>
    </div>
  );
};

export default App; 