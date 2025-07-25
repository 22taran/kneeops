import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatBytes(bytes: number, decimals = 2) {
  if (bytes === 0) return '0 Bytes'
  
  const k = 1024
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
  
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i]
}

export function getConfidenceColor(confidence: number) {
  if (confidence >= 0.7) return 'text-green-600 dark:text-green-400'
  if (confidence >= 0.5) return 'text-yellow-600 dark:text-yellow-400'
  return 'text-red-600 dark:text-red-400'
}

export function getSeverityColor(severity: string) {
  switch (severity.toLowerCase()) {
    case 'high':
      return 'text-red-600 dark:text-red-400'
    case 'medium':
      return 'text-yellow-600 dark:text-yellow-400'
    case 'low':
      return 'text-green-600 dark:text-green-400'
    default:
      return 'text-gray-600 dark:text-gray-400'
  }
}

export function normalizeMRI(data: any): number[] {
  // Handle different input formats
  if (Array.isArray(data)) {
    // Flatten nested arrays and ensure all elements are numbers
    return data.flat(2).map(Number)
  } else if (data?.data) {
    // Handle objects with data property
    const value = Array.isArray(data.data) ? data.data.flat(2) : [data.data]
    return value.map(Number)
  } else if (data && typeof data === 'object') {
    // Convert object values to array and ensure all elements are numbers
    return Object.values(data).flat(2).map(Number)
  }
  
  // Default case - convert to number and wrap in array
  return [Number(data)]
}

export function getMiddleSlice(data: any[] | any, depth = 0): any {
  if (!Array.isArray(data)) return data
  
  if (depth === 2) {
    return data[Math.floor(data.length / 2)]
  }
  
  return getMiddleSlice(data[Math.floor(data.length / 2)], depth + 1)
}

export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout
  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      clearTimeout(timeout)
      func(...args)
    }
    clearTimeout(timeout)
    timeout = setTimeout(later, wait)
  }
}
