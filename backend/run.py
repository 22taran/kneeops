#!/usr/bin/env python3
"""
KneeOps Backend Runner
"""

import uvicorn

if __name__ == "__main__":
    print("🚀 Starting KneeOps Backend...")
    print("   Port: 8000")
    print("   URL: http://localhost:8000")
    print("   Docs: http://localhost:8000/docs")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 