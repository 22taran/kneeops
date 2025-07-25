import React, { useState, useEffect } from 'react';

interface ImageViewerProps {
  file: File | null;
  currentImageIndex: number;
  onImageChange: (index: number) => void;
}

const ImageViewer: React.FC<ImageViewerProps> = ({ file, currentImageIndex, onImageChange }) => {
  const [images, setImages] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!file) {
      setImages([]);
      return;
    }

    setLoading(true);
    
    // Convert .pck file to displayable images
    const reader = new FileReader();
    reader.onload = async (e) => {
      try {
        // Note: This is a simplified version. In a real implementation,
        // you'd need to properly decode the pickle data and convert to images
        // const arrayBuffer = e.target?.result as ArrayBuffer;
        
        // For now, we'll create placeholder images
        // In a real implementation, you'd parse the pickle data and convert to base64
        const placeholderImages = Array.from({ length: 5 }, (_, i) => 
          `data:image/svg+xml;base64,${btoa(`
            <svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
              <rect width="200" height="200" fill="#f0f0f0"/>
              <text x="100" y="100" text-anchor="middle" fill="#666">MRI Slice ${i + 1}</text>
            </svg>
          `)}`
        );
        
        setImages(placeholderImages);
      } catch (error) {
        console.error('Error loading images:', error);
        setImages([]);
      } finally {
        setLoading(false);
      }
    };
    
    reader.readAsArrayBuffer(file);
  }, [file]);

  if (!file) {
    return (
      <div className="bg-gray-100 rounded-lg p-8 text-center">
        <p className="text-gray-500">No file selected</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="bg-gray-100 rounded-lg p-8 text-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto"></div>
        <p className="text-gray-500 mt-2">Loading images...</p>
      </div>
    );
  }

  if (images.length === 0) {
    return (
      <div className="bg-gray-100 rounded-lg p-8 text-center">
        <p className="text-gray-500">No images found in file</p>
      </div>
    );
  }

  return (
    <div className="bg-white border rounded-lg p-4">
      <h3 className="text-lg font-semibold mb-4">MRI Images ({images.length} slices)</h3>
      
      {/* Image Navigation */}
      <div className="flex justify-between items-center mb-4">
        <button
          onClick={() => onImageChange(Math.max(0, currentImageIndex - 1))}
          disabled={currentImageIndex === 0}
          className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50"
        >
          Previous
        </button>
        
        <span className="text-sm text-gray-600">
          {currentImageIndex + 1} of {images.length}
        </span>
        
        <button
          onClick={() => onImageChange(Math.min(images.length - 1, currentImageIndex + 1))}
          disabled={currentImageIndex === images.length - 1}
          className="px-3 py-1 bg-gray-200 rounded disabled:opacity-50"
        >
          Next
        </button>
      </div>
      
      {/* Current Image */}
      <div className="flex justify-center">
        <img
          src={images[currentImageIndex]}
          alt={`MRI Slice ${currentImageIndex + 1}`}
          className="max-w-full h-64 object-contain border rounded"
        />
      </div>
      
      {/* Thumbnail Navigation */}
      <div className="flex justify-center mt-4 space-x-2">
        {images.map((_, index) => (
          <button
            key={index}
            onClick={() => onImageChange(index)}
            className={`w-12 h-12 border rounded ${
              index === currentImageIndex 
                ? 'border-blue-500 bg-blue-50' 
                : 'border-gray-300'
            }`}
          >
            <img
              src={images[index]}
              alt={`Thumbnail ${index + 1}`}
              className="w-full h-full object-cover rounded"
            />
          </button>
        ))}
      </div>
    </div>
  );
};

export default ImageViewer; 