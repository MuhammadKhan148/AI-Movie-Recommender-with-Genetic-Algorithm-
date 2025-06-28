"""
FastAPI Backend for Emotion-Driven Movie Recommender with Genetic Algorithms
FIXED VERSION - Complete integration with enhanced features
"""
# Add this after "import enhanced_genetic_recommender"
from metrics_monitor import MetricsCollector
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import time
import uuid
import logging

# Import the enhanced module
try:
    import enhanced_genetic_recommender
except ImportError:
    print("Error: enhanced_genetic_recommender module not found!")
    import sys
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define models for request/response
class ChatRequest(BaseModel):
    """Model for chat request"""
    user_id: str
    message: str
    
class Movie(BaseModel):
    """Model for movie data"""
    title: str
    genre: str
    description: str
    score: float
    
class ChatResponse(BaseModel):
    """Model for chat response"""
    reply: str
    emotion: str
    confidence: float
    movies: List[Movie] = []
    debug_info: Optional[Dict[str, Any]] = None

class StatsResponse(BaseModel):
    """Model for stats response"""
    users: int
    total_turns: int
    average_satisfaction: float
    ga_stats: Dict[str, Any]

# Create FastAPI app
app = FastAPI(
    title="Enhanced Emotion-Driven Movie Recommender API",
    description="An enhanced API for an emotion-aware movie recommendation chatbot with MovieLens data",
    version="2.0.0"
)
metrics_collector = MetricsCollector()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {"message": "Enhanced Emotion-Driven Movie Recommender API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a chat message and return a response
    
    Args:
        request: ChatRequest containing user_id and message
        
    Returns:
        ChatResponse containing the reply and any movie recommendations
    """
    try:
        # Input validation
        if not request.message or not request.message.strip():
            return ChatResponse(
                reply="I didn't catch that. Could you please say something?",
                emotion="neutral",
                confidence=0.5,
                movies=[],
                debug_info={"error": "empty_message"}
            )
            
        # Truncate extremely long messages
        message = request.message
        if len(message) > 1000:
            message = message[:1000] + "..."
            
        # Process the message using enhanced system
        response = enhanced_genetic_recommender.process_message(request.user_id, message)
        
        # Format movies
        movies = [
            Movie(
                title=movie.get("title", "Unknown"),
                genre=movie.get("genre", "Unknown"),
                description=movie.get("description", ""),
                score=movie.get("score", 0.5)
            )
            for movie in response.get("movies", [])
        ]
        
        # Create response
        return ChatResponse(
            reply=response["reply"],
            emotion=response["emotion"],
            confidence=response["confidence"],
            movies=movies,
            debug_info=response.get("debug_info", {})
        )
    except Exception as e:
        logger.error(f"Error in /chat: {str(e)}", exc_info=True)
        return ChatResponse(
            reply=f"I'm having trouble processing your request. Please try again.",
            emotion="neutral",
            confidence=0.5,
            movies=[],
            debug_info={"error": str(e)}
        )

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get statistics about the chatbot and genetic algorithm
    
    Returns:
        StatsResponse containing stats about users, turns, and GA performance
    """
    try:
        stats = enhanced_genetic_recommender.get_stats()
        
        # Ensure all required fields are present
        required_fields = {
            "users": 0,
            "total_turns": 0,
            "average_satisfaction": 0.0,
            "ga_stats": {}
        }
        
        # Fill in missing fields
        for field, default_value in required_fields.items():
            if field not in stats:
                stats[field] = default_value
        
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.post("/generate-user-id")
async def generate_user_id(name: Optional[str] = None):
    """
    Generate a user ID for demo purposes
    
    Args:
        name: Optional name for the user
        
    Returns:
        Dict containing the generated user_id
    """
    user_id = str(uuid.uuid4())
    return {"user_id": user_id}

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0",
        "components": {
            "enhanced_genetic_recommender": "ok",
            "ml_emotion_detector": "ok" if enhanced_genetic_recommender.USE_ML_EMOTIONS else "not_available"
        }
    }
    
    try:
        # Test that the conversation manager is accessible
        stats = enhanced_genetic_recommender.get_stats()
        health_status["components"]["database"] = "ok"
        health_status["database_info"] = f"Loaded {stats.get('users', 0)} users"
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["database"] = "error"
        health_status["errors"] = str(e)
    
    return health_status

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "debug": str(exc) if app.debug else None
        }
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("Starting Enhanced Emotion-Driven Movie Recommender API")
    logger.info(f"ML Emotion Detection: {'Enabled' if enhanced_genetic_recommender.USE_ML_EMOTIONS else 'Disabled'}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down Enhanced Emotion-Driven Movie Recommender API")

if __name__ == "__main__":
    # Run the server using uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)