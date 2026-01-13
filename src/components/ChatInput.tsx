import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2 } from 'lucide-react';
import { ChatInputProps } from '../types';

const ChatInput: React.FC<ChatInputProps> = ({ 
  onSendMessage, 
  disabled = false, 
  placeholder = "Ask about the MRI analysis..." 
}) => {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = () => {
    const trimmedMessage = message.trim();
    if (trimmedMessage && !disabled) {
      onSendMessage(trimmedMessage);
      setMessage('');
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleTextareaChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setMessage(e.target.value);
  };

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [message]);

  return (
    <div className="border-t border-medical-200 dark:border-medical-700 bg-white dark:bg-medical-800 p-5 backdrop-blur-sm">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-end space-x-3">
          <div className="flex-1 relative">
            <textarea
              ref={textareaRef}
              value={message}
              onChange={handleTextareaChange}
              onKeyPress={handleKeyPress}
              placeholder={placeholder}
              disabled={disabled}
              className="w-full px-4 py-3.5 pr-12 bg-medical-50 dark:bg-medical-900/50 border border-medical-200 dark:border-medical-700 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none min-h-[52px] max-h-32 text-medical-900 dark:text-medical-100 placeholder-medical-400 dark:placeholder-medical-500 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
              rows={1}
            />
          </div>
          
          <button
            onClick={handleSubmit}
            disabled={disabled || !message.trim()}
            className="flex-shrink-0 h-[52px] w-[52px] p-0 flex items-center justify-center rounded-xl bg-gradient-to-r from-primary-600 to-primary-700 hover:from-primary-700 hover:to-primary-800 dark:from-primary-600 dark:to-primary-700 dark:hover:from-primary-700 dark:hover:to-primary-800 text-white font-medium transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl hover:shadow-primary-500/30 hover:scale-105 active:scale-95 disabled:hover:scale-100 disabled:hover:shadow-lg"
          >
            {disabled ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Send className="w-5 h-5" />
            )}
          </button>
        </div>
        
        <div className="mt-3 text-xs text-medical-500 dark:text-medical-400 text-center">
          Press <kbd className="px-2 py-1 bg-medical-100 dark:bg-medical-800 rounded text-xs font-mono">Enter</kbd> to send, <kbd className="px-2 py-1 bg-medical-100 dark:bg-medical-800 rounded text-xs font-mono">Shift+Enter</kbd> for new line
        </div>
      </div>
    </div>
  );
};

export default ChatInput; 