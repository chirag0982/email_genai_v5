"""
Hybrid Flask + FastAPI Application Entry Point
Runs both Flask (for frontend) and FastAPI (for advanced API) on different ports
"""
import threading
import time
import uvicorn
from app import app, socketio
import routes  # noqa: F401
import websocket_handler  # noqa: F401
from fastapi_service import fastapi_app

def run_flask():
    """Run Flask application with SocketIO"""
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, use_reloader=False, log_output=True)

def run_fastapi():
    """Run FastAPI application"""
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="info")

if __name__ == "__main__":
    print("🚀 Starting AI Email Assistant with LangChain and FastAPI integration...")
    print("📧 Flask Frontend: http://localhost:5000")
    print("⚡ FastAPI Backend: http://localhost:8000")
    print("📚 API Documentation: http://localhost:8000/docs")
    
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    # Give Flask a moment to start
    time.sleep(2)
    
    # Start FastAPI in main thread
    run_fastapi()