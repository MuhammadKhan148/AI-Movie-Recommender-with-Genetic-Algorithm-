"""
Enhanced FastAPI Backend with MovieLens Data Integration
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import uvicorn
import time
import uuid

# Import the enhanced module
import enhanced_genetic_recommender

# Define models for request/response
class ChatRequest(BaseModel):
    user_id: str
    message: str
    
class Movie(BaseModel):
    title: str
    genre: str
    description: str
    score: float
    
class ChatResponse(BaseModel):
    reply: str
    emotion: str
    confidence: float
    movies: List[Movie] = []
    debug_info: Optional[Dict[str, Any]] = None

class StatsResponse(BaseModel):
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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Enhanced Emotion-Driven Movie Recommender API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        if not request.message or not request.message.strip():
            return ChatResponse(
                reply="I didn't catch that. Could you please say something?",
                emotion="neutral",
                confidence=0.5,
                movies=[],
                debug_info={"error": "empty_message"}
            )
            
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
    try:
        stats = enhanced_genetic_recommender.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@app.post("/generate-user-id")
async def generate_user_id(name: Optional[str] = None):
    user_id = str(uuid.uuid4())
    return {"user_id": user_id}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "2.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)