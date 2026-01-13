import React from 'react';
import { User, Bot, Loader2 } from 'lucide-react';
import { ChatMessage as ChatMessageType } from '../types';
import AnalysisResults from './AnalysisResults';

interface ChatMessageProps {
  message: ChatMessageType;
}

const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`chat-message flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`flex items-start space-x-3 max-w-[85%] sm:max-w-3xl ${isUser ? 'flex-row-reverse space-x-reverse' : ''}`}>
        <div className={`flex-shrink-0 w-10 h-10 rounded-xl flex items-center justify-center shadow-md transition-transform duration-200 ${
          isUser 
            ? 'bg-gradient-to-br from-primary-500 to-primary-600' 
            : 'bg-gradient-to-br from-medical-100 to-medical-200 dark:from-medical-700 dark:to-medical-600'
        }`}>
          {isUser ? (
            <User className="w-5 h-5 text-white" />
          ) : (
            <Bot className="w-5 h-5 text-medical-700 dark:text-medical-200" />
          )}
        </div>
        
        <div className={`flex-1 ${isUser ? 'text-right' : ''}`}>
          <div className={`inline-block px-5 py-3.5 rounded-2xl shadow-sm transition-all duration-200 ${
            isUser 
              ? 'bg-gradient-to-br from-primary-600 to-primary-700 text-white rounded-br-sm' 
              : 'bg-white dark:bg-medical-800 border border-medical-200 dark:border-medical-700 text-medical-900 dark:text-medical-100 rounded-bl-sm'
          }`}>
            {message.isLoading ? (
              <div className="flex items-center space-x-3">
                <Loader2 className="w-5 h-5 animate-spin" />
                <span className="text-sm font-medium">AI is analyzing...</span>
              </div>
            ) : (
              <div className="text-sm leading-relaxed">
                {message.analysisResults ? (
                  <AnalysisResults
                    averageConfidence={message.analysisResults.averageConfidence}
                    diagnosis={message.analysisResults.diagnosis}
                    diagnosisConfidence={message.analysisResults.diagnosisConfidence}
                    processedImages={message.analysisResults.processedImages}
                  />
                ) : (
                  <div className="whitespace-pre-wrap prose prose-sm dark:prose-invert max-w-none">
                    {message.content}
                  </div>
                )}
              </div>
            )}
          </div>
          
          <div className={`text-xs text-medical-500 dark:text-medical-400 mt-2 px-1 ${
            isUser ? 'text-right' : 'text-left'
          }`}>
            {message.timestamp.toLocaleTimeString([], { 
              hour: '2-digit', 
              minute: '2-digit' 
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChatMessage; 