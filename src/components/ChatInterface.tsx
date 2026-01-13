import React, { useRef, useEffect } from 'react';
import ChatMessage from './ChatMessage';
import ChatInput from './ChatInput';
import { ChatMessage as ChatMessageType } from '../types';

interface ChatInterfaceProps {
  messages: ChatMessageType[];
  onSendMessage: (message: string) => void;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({
  messages,
  onSendMessage,
  disabled = false,
  placeholder,
  className = ''
}) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className={`flex flex-col h-full max-h-full bg-white dark:bg-medical-800 overflow-hidden ${className}`}>
      {/* Chat Messages Area - Fixed height with scroll */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden p-6 space-y-5 scroll-smooth min-h-0 max-h-full">
        {messages.length === 0 ? (
          <div className="text-center py-12">
            <div className="max-w-lg mx-auto animate-fade-in">
              <div className="bg-gradient-to-br from-primary-50 via-white to-medical-50 dark:from-medical-800/50 dark:via-medical-800/30 dark:to-medical-900/50 rounded-2xl p-8 border border-medical-200 dark:border-medical-700 shadow-lg">
                <div className="w-16 h-16 bg-gradient-to-br from-primary-500 to-primary-600 rounded-2xl flex items-center justify-center mx-auto mb-4 shadow-lg shadow-primary-500/20">
                  <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                  </svg>
                </div>
                <h3 className="text-2xl font-bold text-medical-900 dark:text-white mb-3">
                  Welcome to KneeOps AI
                </h3>
                <p className="text-medical-600 dark:text-medical-400 text-sm mb-6">
                  Upload an MRI file to start analyzing ACL injuries. You can ask questions like:
                </p>
                <ul className="text-left text-sm text-medical-700 dark:text-medical-300 space-y-2.5">
                  <li className="flex items-start">
                    <span className="text-primary-500 mr-2 font-bold">•</span>
                    <span>"What are the key findings in this MRI?"</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-primary-500 mr-2 font-bold">•</span>
                    <span>"Is there evidence of ACL injury?"</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-primary-500 mr-2 font-bold">•</span>
                    <span>"What's the severity of the damage?"</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-primary-500 mr-2 font-bold">•</span>
                    <span>"Are there any other knee injuries present?"</span>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-5">
            {messages.map((message, index) => (
              <div key={message.id} className="animate-slide-up" style={{ animationDelay: `${index * 0.1}s` }}>
                <ChatMessage message={message} />
              </div>
            ))}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Chat Input - Fixed at bottom */}
      <div className="flex-shrink-0">
        <ChatInput
          onSendMessage={onSendMessage}
          disabled={disabled}
          placeholder={placeholder}
        />
      </div>
    </div>
  );
};

export default ChatInterface; 