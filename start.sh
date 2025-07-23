#!/bin/bash

echo "🚀 Starting KneeOps Application..."

# Start Backend
echo "📦 Starting Backend..."
cd backend
source venv/bin/activate
python run.py &
BACKEND_PID=$!
cd ..

# Wait a moment for backend to start
sleep 3

# Start Frontend
echo "📦 Starting Frontend..."
npm start &
FRONTEND_PID=$!

echo "✅ Both services started!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for user to stop
wait 