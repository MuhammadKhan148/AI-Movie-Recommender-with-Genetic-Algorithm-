"""
FastAPI Backend for Emotion-Driven Movie Recommender with Genetic Algorithms

This module provides a RESTful API for the emotion-aware chatbot using FastAPI.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import time
import uuid

# Import the chatbot module
import genetic_recommender

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
    title="Emotion-Driven Movie Recommender API",
    description="An API for an emotion-aware movie recommendation chatbot with genetic algorithm optimization",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# User ID mapping for demo purposes
user_id_mapping = {}

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {"message": "Emotion-Driven Movie Recommender API is running"}

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
            
        # Process the message
        response = genetic_recommender.process_message(request.user_id, message)
        
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
        print(f"Error in /chat: {str(e)}")
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
        stats = genetic_recommender.get_stats()
        return stats
    except Exception as e:
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
    if name:
        user_id_mapping[user_id] = name
    return {"user_id": user_id}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # Run the server using uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)