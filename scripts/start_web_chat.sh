#!/bin/bash
# RFSN Web Chat UI Startup Script
# Starts all required servers for the web chat interface

set -e

echo "🚀 Starting RFSN Web Chat UI..."
echo "================================"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Kill any existing processes
echo -e "${YELLOW}Stopping any existing servers...${NC}"
pkill -f "uvicorn orchestrator" 2>/dev/null || true
pkill -f "web_chat_ui/backend/main.py" 2>/dev/null || true
sleep 1

# 1. Start Orchestrator (main API on port 8000)
echo -e "${GREEN}[1/3] Starting Orchestrator on port 8000...${NC}"
cd "$PROJECT_ROOT/Python"
python -m uvicorn orchestrator:app --host 0.0.0.0 --port 8000 &
ORCHESTRATOR_PID=$!
sleep 3

# 2. Start Web Chat Backend (proxy on port 3001)
echo -e "${GREEN}[2/3] Starting Web Chat Backend on port 3001...${NC}"
cd "$PROJECT_ROOT/web_chat_ui/backend"
python main.py &
BACKEND_PID=$!
sleep 2

# 3. Start Frontend (Vite dev server on port 5173)
echo -e "${GREEN}[3/3] Starting Frontend on port 5173...${NC}"
cd "$PROJECT_ROOT/web_chat_ui/frontend"
npm run dev &
FRONTEND_PID=$!
sleep 3

echo ""
echo "================================"
echo -e "${GREEN}✅ All servers started!${NC}"
echo ""
echo "  📡 Orchestrator API:  http://localhost:8000"
echo "  🔌 Chat Backend:      http://localhost:3001"
echo "  🌐 Web UI:            http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop all servers"
echo "================================"

# Wait for any process to exit
wait
