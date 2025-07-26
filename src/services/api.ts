// API Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Types for API communication
export interface MLResponse {
  analysis: {
    acl_status: 'intact' | 'partial_tear' | 'complete_tear' | 'unknown';
    severity_grade: 'grade_0' | 'grade_1' | 'grade_2' | 'grade_3' | 'unknown';
    confidence_score: number;
    findings: string[];
    recommendations: string[];
  };
  message: string;
  success: boolean;
}

export interface ChatResponse {
  response: string;
  analysis_context?: {
    acl_status: string;
    severity: string;
    confidence: number;
  };
  success: boolean;
}

// API Service Class
class APIService {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  // Upload MRI file to ML model
  async uploadMRI(file: File): Promise<any> {
    console.log('Starting file upload for:', file.name, 'size:', file.size, 'type:', file.type);
    const formData = new FormData();
    formData.append('file', file);

    try {
      console.log('Sending request to:', `${this.baseURL}/api/upload-mri`);
      console.time('MRI Upload');
      const response = await fetch(`${this.baseURL}/api/upload-mri`, {
        method: 'POST',
        body: formData,
      });
      console.timeEnd('MRI Upload');

      if (!response.ok) {
        const errorText = await response.text();
        console.error('HTTP error response:', response.status, response.statusText, errorText);
        throw new Error(`HTTP error! status: ${response.status}, response: ${errorText}`);
      }

      const result = await response.json();
      console.log('Upload successful, response keys:', Object.keys(result));
      
      // Log MRI data structure if it exists
      if (result.mri_data) {
        console.log('MRI data type:', Array.isArray(result.mri_data) ? 'array' : typeof result.mri_data);
        if (Array.isArray(result.mri_data)) {
          console.log('MRI data length:', result.mri_data.length);
          if (result.mri_data.length > 0) {
            console.log('First slice type:', typeof result.mri_data[0]);
            if (Array.isArray(result.mri_data[0])) {
              console.log('First slice dimensions:', result.mri_data[0].length, 'x', 
                Array.isArray(result.mri_data[0][0]) ? result.mri_data[0][0].length : '1D');
            }
          }
        }
      }
      
      // Handle the new detailed response format
      if (result.success) {
        return result; // Return the full detailed response
      } else {
        throw new Error('Backend returned unsuccessful response');
      }
    } catch (error) {
      console.error('Error uploading MRI:', error);
      throw error;
    }
  }

  // Send chat message to ML model
  async sendChatMessage(message: string, fileId: string): Promise<ChatResponse> {
    try {
      const response = await fetch(`${this.baseURL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message, file_id: fileId }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Error sending chat message:', error);
      throw error;
    }
  }

  // Get analysis status
  async getAnalysisStatus(fileId: string): Promise<{ status: 'processing' | 'completed' | 'failed' }> {
    return { status: 'completed' };
  }

  // Get detailed analysis results
  async getAnalysisResults(fileId: string): Promise<MLResponse> {
    return {
      success: true,
      message: "Analysis completed",
      analysis: {
        acl_status: 'unknown',
        severity_grade: 'unknown',
        confidence_score: 0.0,
        findings: ["Analysis completed successfully"],
        recommendations: ["Consult with a medical professional"]
      }
    };
  }

  // Helper method to map prediction to ACL status
  private _mapACLStatus(prediction: any): 'intact' | 'partial_tear' | 'complete_tear' | 'unknown' {
    if (!prediction) return 'unknown';
    
    const predictionStr = String(prediction);
    const predictionLower = predictionStr.toLowerCase();
    
    if (predictionLower.includes('healthy') || predictionLower.includes('normal')) {
      return 'intact';
    } else if (predictionLower.includes('acl injury')) {
      return 'partial_tear';
    } else if (predictionLower.includes('meniscus tear')) {
      return 'complete_tear';
    } else {
      return 'unknown';
    }
  }

  // Helper method to map prediction to severity grade
  private _mapSeverity(prediction: any): 'grade_0' | 'grade_1' | 'grade_2' | 'grade_3' | 'unknown' {
    if (!prediction) return 'unknown';
    
    const predictionStr = String(prediction);
    const predictionLower = predictionStr.toLowerCase();
    
    if (predictionLower.includes('normal') || predictionLower.includes('intact') || predictionLower.includes('healthy')) {
      return 'grade_0';
    } else if (predictionLower.includes('mild') || predictionLower.includes('partial') || predictionLower.includes('acl injury')) {
      return 'grade_1';
    } else if (predictionLower.includes('severe') || predictionLower.includes('complete') || predictionLower.includes('meniscus tear')) {
      return 'grade_3';
    } else {
      return 'unknown';
    }
  }
}

const apiService = new APIService();
export default apiService; 