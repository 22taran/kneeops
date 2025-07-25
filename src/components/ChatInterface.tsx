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
    <div className={`flex flex-col h-full bg-white dark:bg-medical-800 ${className}`}>
      {/* Chat Messages Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center py-8">
            <div className="max-w-md mx-auto">
              <div className="bg-medical-50 rounded-lg p-6 border border-medical-200">
                <h3 className="text-lg font-medium text-medical-900 mb-2">
                  Welcome to KneeOps AI
                </h3>
                <p className="text-medical-600 text-sm mb-4">
                  Upload an MRI file to start analyzing ACL injuries. You can ask questions like:
                </p>
                <ul className="text-left text-sm text-medical-600 space-y-1">
                  <li>• "What are the key findings in this MRI?"</li>
                  <li>• "Is there evidence of ACL injury?"</li>
                  <li>• "What's the severity of the damage?"</li>
                  <li>• "Are there any other knee injuries present?"</li>
                </ul>
              </div>
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {messages.map((message) => (
              <ChatMessage key={message.id} message={message} />
            ))}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Chat Input */}
      <ChatInput
        onSendMessage={onSendMessage}
        disabled={disabled}
        placeholder={placeholder}
      />
    </div>
  );
};

export default ChatInterface; 