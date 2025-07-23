#!/bin/bash

echo "🚀 Setting up KneeOps Application..."

# Backend Setup
echo "📦 Setting up Backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Make activate script executable
chmod +x venv/bin/activate

echo "✅ Backend setup complete!"

# Frontend Setup
echo "📦 Setting up Frontend..."
cd ..

# Install frontend dependencies
echo "Installing Node.js dependencies..."
npm install

echo "✅ Frontend setup complete!"

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To start the application:"
echo "1. Backend: cd backend && source venv/bin/activate && python run.py"
echo "2. Frontend: npm start"
echo ""
echo "Or use the start script: ./start.sh" 