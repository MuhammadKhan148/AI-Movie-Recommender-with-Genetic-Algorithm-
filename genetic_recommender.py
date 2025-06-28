"""
Emotion-Driven Movie Recommender with Genetic Algorithm Optimization

This module provides the core AI components for the movie recommendation system:
1. A genetic algorithm for optimizing conversation paths
2. An emotion-aware conversation manager
3. A simple emotion detection system

The genetic algorithm continuously evolves conversation strategies based on user satisfaction,
demonstrating an advanced AI optimization technique as required by the project rubric.
"""

import random
import time
import re
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

try:
    from ml_emotion_detector import MLEmotionDetectorOptional
    USE_ML_EMOTIONS = True
    print("ML emotion detection available")
except ImportError:
    USE_ML_EMOTIONS = False
    print("Using simple keyword-based emotion detection")

# ======================================================
# Genetic Algorithm for Conversation Path Optimization
# ======================================================

class GeneticAlgorithm:
    """
    Genetic Algorithm implementation for optimizing conversation paths.
    
    This class handles the evolutionary process including:
    - Selection of the fittest individuals (conversation paths)
    - Crossover to create new offspring
    - Mutation to maintain diversity
    - Fitness evaluation based on user responses
    
    The algorithm continuously evolves to find the most effective conversation
    strategies for recommending movies based on user emotions.
    """
    
    def __init__(self, population_size: int = 10, mutation_rate: float = 0.2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation = 0
        
        # Available stages for conversation paths
        self.available_stages = [
            "greeting",
            "emotion_elicitation",
            "genre_preference",
            "specific_preference",
            "recommendation",
            "feedback",
            "refinement"
        ]
        
        # Initialize population with random conversation paths
        self.population = self._initialize_population()
        
        # Track fitness of each path
        self.fitness_scores = {}
        
        # Current best path
        self.best_path_index = 0
        self.best_path = self.population[0]
        
        # Performance metrics for reporting
        self.average_fitness = 0.0
        self.best_fitness = 0.0
        
    def _initialize_population(self) -> List[List[str]]:
        """Initialize a population of valid conversation paths"""
        population = []
        
        # Always include the default path as first member
        default_path = [
            "greeting",
            "emotion_elicitation",
            "genre_preference",
            "specific_preference",
            "recommendation",
            "feedback",
            "refinement"
        ]
        population.append(default_path)
        
        # Generate additional random paths
        for _ in range(self.population_size - 1):
            # All paths must start with greeting
            path = ["greeting"]
            
            # Add 2-5 intermediate stages
            intermediate_stages = random.sample(
                self.available_stages[1:-1],  # Exclude greeting and refinement
                random.randint(2, 5)  # Random number of stages
            )
            path.extend(intermediate_stages)
            
            # All paths should eventually reach recommendation
            if "recommendation" not in path:
                path.append("recommendation")
                
            # End with refinement
            if path[-1] != "refinement":
                path.append("refinement")
                
            population.append(path)
            
        return population
    
    def select_parents(self) -> List[List[str]]:
        """Tournament selection of parents based on fitness"""
        if not self.fitness_scores:
            # If no fitness data yet, return random parents
            return random.sample(self.population, 2)
        
        # Sort paths by fitness
        sorted_paths = sorted(
            self.fitness_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Select top 2 performing paths
        best_indices = [idx for idx, _ in sorted_paths[:2]]
        return [self.population[idx] for idx in best_indices]
    
    def crossover(self, parents: List[List[str]]) -> List[List[str]]:
        """Perform single-point crossover between parents"""
        if len(parents) < 2:
            return parents
            
        parent1, parent2 = parents
        min_len = min(len(parent1), len(parent2))
        
        # Ensure minimum path length for crossover
        if min_len <= 2:
            return parents
            
        # Select crossover point (not at start or end)
        crossover_point = random.randint(1, min_len - 1)
        
        # Create offspring
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        # Ensure valid paths (must include recommendation stage)
        for child in [child1, child2]:
            if "recommendation" not in child:
                position = random.randint(1, len(child) - 1)
                child.insert(position, "recommendation")
                
            # Ensure refinement is at the end
            if child[-1] != "refinement":
                child.append("refinement")
                
        return [child1, child2]
    
    def mutate(self, offspring: List[List[str]]) -> List[List[str]]:
        """Apply mutation to offspring to maintain genetic diversity"""
        mutated = []
        
        for path in offspring:
            # Deep copy the path
            mutated_path = path.copy()
            
            # Apply mutation with probability
            if random.random() < self.mutation_rate:
                # Select random position to mutate (not first or last)
                if len(mutated_path) > 2:
                    position = random.randint(1, len(mutated_path) - 2)
                    
                    # Get current stage to replace
                    current = mutated_path[position]
                    
                    # Get valid replacements (different from current)
                    options = [s for s in self.available_stages 
                              if s != current and s != "greeting"]
                    
                    # Replace with a different stage
                    mutated_path[position] = random.choice(options)
            
            mutated.append(mutated_path)
            
        return mutated
    
    def evolve(self) -> None:
        """Perform one generation of evolution"""
        # Only evolve if we have fitness data
        if not self.fitness_scores:
            return
            
        # Select parents
        parents = self.select_parents()
        
        # Create offspring through crossover
        offspring = self.crossover(parents)
        
        # Apply mutation
        mutated_offspring = self.mutate(offspring)
        
        # Replace worst individuals in population
        if self.fitness_scores:
            # Sort by fitness (ascending)
            sorted_indices = [idx for idx, _ in 
                             sorted(self.fitness_scores.items(), 
                                    key=lambda x: x[1])]
            
            # Replace worst performers
            for i, idx in enumerate(sorted_indices):
                if i < len(mutated_offspring):
                    self.population[idx] = mutated_offspring[i]
                    
        # Increment generation counter
        self.generation += 1
        
        # Update best path
        if self.fitness_scores:
            best_idx = max(self.fitness_scores.items(), key=lambda x: x[1])[0]
            self.best_path_index = best_idx
            self.best_path = self.population[best_idx]
            self.best_fitness = self.fitness_scores[best_idx]
            
            # Calculate average fitness
            self.average_fitness = sum(self.fitness_scores.values()) / len(self.fitness_scores)
    
    def update_fitness(self, path_index: int, reward: float) -> None:
        """Update fitness score for a path"""
        if path_index not in self.fitness_scores:
            self.fitness_scores[path_index] = 0
            
        self.fitness_scores[path_index] += reward
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the genetic algorithm"""
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_path": self.best_path,
            "best_fitness": self.best_fitness,
            "average_fitness": self.average_fitness,
            "mutation_rate": self.mutation_rate,
            "population": self.population[:3],  # Return first 3 paths for visualization
            "fitness_scores": dict(sorted(self.fitness_scores.items())[:5]) if self.fitness_scores else {}
        }

# ======================================================
# Emotion-Aware Conversation Manager with Loop Prevention
# ======================================================

class EmotionDrivenConversationManager:
    """
    Manages conversation flow based on emotion detection and GA optimization.
    
    Key features:
    - Proactive, question-driven conversation flow
    - Loop detection and prevention
    - Integration with genetic algorithm for path optimization
    - Emotion-to-genre mapping for personalized recommendations
    """
    
    def __init__(self):
        if USE_ML_EMOTIONS:
            self.emotion_detector = MLEmotionDetectorOptional()  # Use ML if available
        else:
            self.emotion_detector = EmotionDetector()  # Keep your original
        
        # Initialize genetic algorithm for conversation optimization
        self.ga = GeneticAlgorithm()
        
        # Track current state for each user
        self.user_states = {}
        
        # Question templates for each stage
        self.questions = self._initialize_questions()
        
        # Emotion-to-genre mapping
        self.emotion_genre_mapping = self._initialize_emotion_mapping()
        
        # Movie database (simplified for demo)
        self.movie_database = MovieDatabase()
        
    def _initialize_questions(self) -> Dict[str, List[str]]:
        """Initialize question templates for each conversation stage"""
        return {
            "greeting": [
                "Hello! I'm your movie recommendation assistant. How are you feeling today?",
                "Hi there! I'm here to find the perfect movie for you. How's your day going?",
                "Welcome! I'd love to help you discover your next favorite film. How are you feeling right now?"
            ],
            "emotion_elicitation": [
                "Before I suggest movies, I'd like to understand your mood better. Would you say you're feeling happy, relaxed, energetic, or perhaps something else?",
                "Movies can really connect with how we're feeling. Would you describe your current mood as upbeat, thoughtful, excited, or something different?",
                "To find the perfect movie match, it helps to know your emotional state. Are you feeling positive, reflective, or looking for an escape today?"
            ],
            "genre_preference": [
                "What genres do you typically enjoy watching? Action, comedy, drama, sci-fi, horror, or something else?",
                "I'd like to understand your taste in movies. Which genres do you find yourself drawn to most often?",
                "Everyone has their favorite types of films. What genres do you usually prefer?"
            ],
            "specific_preference": [
                "Do you have any favorite directors or actors whose work you particularly enjoy?",
                "Are there any specific movies you've watched recently that you really loved?",
                "What elements of storytelling do you value most? Plot twists, character development, visual effects, or something else?"
            ],
            "recommendation": [
                "Based on what you've shared, I think you might enjoy [MOVIE]. It's a [GENRE] film that matches your [EMOTION] mood. Does that sound interesting?",
                "Given your preferences and current mood, I'd recommend [MOVIE]. It's known for its [FEATURE] which aligns with what you're looking for. How does that sound?",
                "I believe [MOVIE] would be perfect for you right now. It has the [QUALITY] you enjoy and works well with your current emotional state. Would you like to know more about it?"
            ],
            "feedback": [
                "How does that recommendation sound to you? Does it match what you were looking for?",
                "Does this suggestion align with the kind of movie experience you're seeking today?",
                "Is this recommendation on the right track, or would you prefer something different?"
            ],
            "refinement": [
                "What aspects of my recommendation appeal to you, and what would you like to change?",
                "To refine my suggestions, could you tell me more about what you're specifically looking for today?",
                "What elements would make a movie perfect for your current mood and preferences?"
            ]
        }
    
    def _initialize_emotion_mapping(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize mapping from emotions to recommended genres"""
        return {
            'joy': {
                'primary': ['Comedy', 'Animation', 'Musical'],
                'secondary': ['Adventure', 'Family', 'Fantasy']
            },
            'sadness': {
                'primary': ['Comedy', 'Animation', 'Adventure'],  # Changed to provide escape
                'secondary': ['Action', 'Fantasy', 'Sci-Fi']     # Genres that help you escape reality
            },
            'anger': {
                'primary': ['Action', 'Thriller'],
                'secondary': ['Crime', 'War', 'Sport']
            },
            'fear': {
                'primary': ['Comedy', 'Animation', 'Fantasy'],
                'secondary': ['Adventure', 'Family', 'Romance']
            },
            'surprise': {
                'primary': ['Mystery', 'Sci-Fi', 'Fantasy'],
                'secondary': ['Thriller', 'Adventure', 'Horror']
            },
            'neutral': {
                'primary': ['Documentary', 'Biography', 'Drama'],
                'secondary': ['Any preferred genres']
            }
        }
    
    def get_user_state(self, user_id: str) -> Dict[str, Any]:
        """Get or initialize user state"""
        if user_id not in self.user_states:
            # Initialize new user with the best conversation path from GA
            self.user_states[user_id] = {
                "current_stage_index": 0,  # Index in the conversation path
                "path": self.ga.best_path.copy(),  # Copy the best path from GA
                "path_index": self.ga.best_path_index,  # Track which GA path we're using
                "detected_emotion": None,
                "emotion_confidence": 0,
                "genres": [],
                "specific_preferences": [],
                "recommended_movies": [],
                "consecutive_yes_count": 0,  # For loop detection
                "satisfaction_score": 0.0,  # For GA fitness
                "turn_count": 0,
                "last_response": ""
            }
            
        return self.user_states[user_id]
    
    def get_next_question(self, user_id: str, emotion: str, confidence: float, message: str) -> Dict[str, Any]:
        """
        Get the next question based on the conversation path.
        Includes loop detection and prevention.
        
        Returns a dictionary with:
        - reply: The bot's response
        - movies: List of recommended movies
        - debug_info: Information for visualization
        """
        state = self.get_user_state(user_id)
        state["turn_count"] += 1
        
        # Update emotion if detected with high confidence
        if confidence > 0.4 and state["detected_emotion"] is None:
            state["detected_emotion"] = emotion
            state["emotion_confidence"] = confidence
        
        # Check for meta-questions about the system
        meta_questions = [
            "how does this work", "what are you", "who made you", 
            "how do you work", "what is this", "what ai"
        ]
        if any(q in message.lower() for q in meta_questions):
            return {
                "reply": "I'm an AI-powered movie recommender that uses genetic algorithms to optimize conversations and emotion detection to personalize recommendations. I learn from user interactions to get better over time. What kind of movies do you enjoy?",
                "movies": [],
                "debug_info": {"meta_question": True, "stage": "explanation"}
            }
            
        # Check for commands
        if message.lower() in ["restart", "start over", "reset"]:
            # Reset user state
            state["current_stage_index"] = 0
            state["detected_emotion"] = None
            state["consecutive_yes_count"] = 0
            state["genres"] = []
            
            return {
                "reply": "Let's start fresh! How are you feeling today?",
                "movies": [],
                "debug_info": {"command": "restart", "stage": "greeting"}
            }
        
        # Check for yes/no response for loop detection
        message_lower = message.lower()
        is_yes_response = any(word in message_lower for word in 
                            ["yes", "yeah", "sure", "okay", "ok", "yep", "yup", "i would"])
        
        # LOOP PREVENTION: Track consecutive yes responses
        if is_yes_response:
            state["consecutive_yes_count"] += 1
        else:
            state["consecutive_yes_count"] = 0
        
        # Force progression if stuck in a loop
        if state["consecutive_yes_count"] > 1:
            print(f"Loop detected for user {user_id} - forcing progression")
            
            # Move to next stage in the path
            if state["current_stage_index"] < len(state["path"]) - 1:
                state["current_stage_index"] += 1
            
            # Reset counter
            state["consecutive_yes_count"] = 0
            
            # Apply negative reinforcement to this path
            self.ga.update_fitness(state["path_index"], -2.0)
        
        # Extract current stage from path
        if state["current_stage_index"] < len(state["path"]):
            current_stage = state["path"][state["current_stage_index"]]
        else:
            # Fallback to recommendation if index out of bounds
            current_stage = "recommendation"
        
        # Skip genre preference if we already have genres
        if current_stage == "genre_preference" and state["genres"]:
            state["current_stage_index"] += 1
            current_stage = state["path"][state["current_stage_index"]]
        
        # Check for explicit genre mentions
        extracted_genres = self._extract_genres(message_lower)
        if extracted_genres:
            state["genres"].extend([g for g in extracted_genres if g not in state["genres"]])
            
            # Apply positive reinforcement - successfully gathered preferences
            self.ga.update_fitness(state["path_index"], 1.0)
            
            # If we have genres and we're in emotion_elicitation, skip to specific_preference
            if current_stage == "emotion_elicitation" and not state["specific_preferences"]:
                # Find index of specific_preference in path
                try:
                    next_idx = state["path"].index("specific_preference")
                    state["current_stage_index"] = next_idx
                    current_stage = "specific_preference"
                except ValueError:
                    # If not in path, move to next stage
                    state["current_stage_index"] += 1
                    if state["current_stage_index"] < len(state["path"]):
                        current_stage = state["path"][state["current_stage_index"]]
        
        # Check for explicit movie mentions for specific preferences
        movie_mentions = self._extract_movie_mentions(message)
        if movie_mentions:
            state["specific_preferences"].extend([m for m in movie_mentions if m not in state["specific_preferences"]])
            
            # Apply positive reinforcement
            self.ga.update_fitness(state["path_index"], 0.5)
        
        # Select question based on current stage
        if current_stage == "greeting":
            response = random.choice(self.questions["greeting"])
            
            # Move to next stage automatically after greeting
            state["current_stage_index"] += 1
            
            return {
                "reply": response,
                "movies": [],
                "debug_info": {"stage": current_stage, "path": state["path"]}
            }
            
        elif current_stage == "emotion_elicitation":
            response = random.choice(self.questions["emotion_elicitation"])
            
            # Only progress if we have emotion or explicit answer
            if state["detected_emotion"] or is_yes_response:
                state["current_stage_index"] += 1
            
            return {
                "reply": response,
                "movies": [],
                "debug_info": {"stage": current_stage, "path": state["path"], "emotion": state["detected_emotion"]}
            }
            
        elif current_stage == "genre_preference":
            response = random.choice(self.questions["genre_preference"])
            
            # Progress if we have genres or explicit answer
            if state["genres"] or is_yes_response:
                state["current_stage_index"] += 1
            
            return {
                "reply": response,
                "movies": [],
                "debug_info": {"stage": current_stage, "path": state["path"], "genres": state["genres"]}
            }
            
        elif current_stage == "specific_preference":
            response = random.choice(self.questions["specific_preference"])
            
            # Always progress after asking about specific preferences
            state["current_stage_index"] += 1
            
            return {
                "reply": response,
                "movies": [],
                "debug_info": {"stage": current_stage, "path": state["path"], "preferences": state["specific_preferences"]}
            }
            
        elif current_stage == "recommendation":
            # Generate movie recommendation
            movie, genre, description = self._generate_recommendation(state)
            state["recommended_movies"].append(movie)
            
            # Fill in placeholders in template
            template = random.choice(self.questions["recommendation"])
            emotion_adj = self._get_emotion_adjective(state["detected_emotion"] or "neutral")
            
            response = template.replace("[MOVIE]", movie)
            response = response.replace("[GENRE]", genre)
            response = response.replace("[EMOTION]", state["detected_emotion"] or "current")
            response = response.replace("[FEATURE]", emotion_adj + " storytelling")
            response = response.replace("[QUALITY]", emotion_adj + " elements")
            
            # Move to next stage
            state["current_stage_index"] += 1
            
            return {
                "reply": response,
                "movies": [{
                    "title": movie, 
                    "genre": genre,
                    "description": description,
                    "score": 0.95
                }],
                "debug_info": {"stage": current_stage, "path": state["path"], "recommendation": movie}
            }
            
        elif current_stage == "feedback":
            response = random.choice(self.questions["feedback"])
            
            # Progress after feedback
            state["current_stage_index"] += 1
            
            # Check for positive feedback
            if any(word in message_lower for word in ["good", "great", "nice", "perfect", "love", "thanks"]):
                # Apply positive reinforcement to this path
                self.ga.update_fitness(state["path_index"], 2.0)
                state["satisfaction_score"] += 1.0
                
            # Check for negative feedback
            elif any(word in message_lower for word in ["bad", "terrible", "not", "don't", "didn't"]):
                # Apply negative reinforcement
                self.ga.update_fitness(state["path_index"], -1.0)
                state["satisfaction_score"] -= 0.5
                
            return {
                "reply": response,
                "movies": [],
                "debug_info": {"stage": current_stage, "path": state["path"]}
            }
            
        elif current_stage == "refinement":
            response = random.choice(self.questions["refinement"])
            
            # After refinement, go back to recommendation
            try:
                recommend_idx = state["path"].index("recommendation")
                state["current_stage_index"] = recommend_idx
            except ValueError:
                # If not in path, just reset to beginning
                state["current_stage_index"] = 0
                
            return {
                "reply": response,
                "movies": [],
                "debug_info": {"stage": current_stage, "path": state["path"]}
            }
            
        else:
            # Fallback
            response = "What kind of movies are you interested in today?"
            state["current_stage_index"] = 0
            
            return {
                "reply": response,
                "movies": [],
                "debug_info": {"stage": "fallback", "path": state["path"]}
            }
    
    def _extract_genres(self, message: str) -> List[str]:
        """Extract movie genres mentioned in message with improved detection"""
        genres = []
        common_genres = [
            "action", "adventure", "comedy", "drama", "horror", 
            "thriller", "sci-fi", "fantasy", "romance", "documentary",
            "animation", "mystery", "crime", "musical", "western",
            "superhero", "history", "war"
        ]
        
        # Check for capitalized genres
        message_words = message.split()
        for word in message_words:
            if word.title() in ["Action", "Comedy", "Drama"]:
                if word.lower() not in genres:
                    genres.append(word.lower())
        
        # Also check for genres in lowercase within the message
        for genre in common_genres:
            if genre in message.lower() and genre not in genres:
                genres.append(genre)
        
        # Handle special cases like "sci-fi"
        if "sci fi" in message.lower() or "science fiction" in message.lower():
            if "sci-fi" not in genres:
                genres.append("sci-fi")
        
        return genres
    
    def _extract_movie_mentions(self, message: str) -> List[str]:
        """
        Extract movie titles mentioned in message.
        This is a simplified version - in a real system this would
        use a more sophisticated entity extraction approach.
        """
        # Simple heuristic - look for text in quotes
        quoted = re.findall(r'"([^"]*)"', message)
        quoted.extend(re.findall(r"'([^']*)'", message))
        
        # Also detect common movie title patterns
        movie_patterns = [
            r"(?:watched|liked|loved|enjoyed|recommended|seen)\s+([A-Z][a-zA-Z0-9\s]+)",
            r"(?:favorite movie is|best movie|great movie)\s+([A-Z][a-zA-Z0-9\s]+)"
        ]
        
        for pattern in movie_patterns:
            matches = re.findall(pattern, message)
            quoted.extend(matches)
            
        # Remove duplicates and return
        return list(set(quoted))
    
    def _generate_recommendation(self, state: Dict[str, Any]) -> Tuple[str, str, str]:
        """Generate movie recommendation based on user state"""
        emotion = state["detected_emotion"] or "neutral"
        genres = state["genres"]
        
        # Use the last mentioned genre
        if genres:
            genre = genres[-1]  # Take the most recent genre
            movie = self.movie_database.get_movie_by_genre_and_emotion(genre, emotion, state["recommended_movies"])
            return movie["title"], genre.title(), movie["description"]
        
        # If no explicit genres mentioned, use emotion-to-genre mapping
        if emotion in self.emotion_genre_mapping:
            primary_genres = [g.lower() for g in self.emotion_genre_mapping[emotion]["primary"]]
            genre = primary_genres[0] if primary_genres else "comedy"
            movie = self.movie_database.get_movie_by_genre_and_emotion(genre, emotion, state["recommended_movies"])
            return movie["title"], genre.title(), movie["description"]
        
        # Fallback
        movie = self.movie_database.get_movie_by_emotion(emotion, state["recommended_movies"])
        
        # Use actual detected emotion in response
        detected_emotion = state["detected_emotion"] or "neutral"
        response = movie["description"].replace("[EMOTION]", detected_emotion)
        
        return movie["title"], movie["genre"].title(), response
    
    def _get_emotion_adjective(self, emotion: str) -> str:
        """Convert emotion to descriptive adjective"""
        emotion_map = {
            "joy": "uplifting",
            "sadness": "reflective",
            "anger": "cathartic",
            "fear": "comforting",
            "surprise": "thought-provoking",
            "neutral": "engaging"
        }
        return emotion_map.get(emotion, "enjoyable")
        
    def process_message(self, user_id: str, message: str) -> Dict[str, Any]:
        """
        Process a message from a user and return a response.
        
        Args:
            user_id: Unique identifier for the user
            message: The user's message
            
        Returns:
            Dict containing the response, any movie recommendations,
            and debug information for visualization
        """
        # Validate input
        if not message or not message.strip():
            return {
                "reply": "I didn't catch that. Could you please say something?",
                "emotion": "neutral",
                "confidence": 0.5,
                "movies": [],
                "debug_info": {"error": "empty_message"}
            }
        
        # Truncate extremely long messages
        if len(message) > 1000:
            message = message[:1000] + "..."
            
        # Get emotion from message
        emotion, confidence = self.emotion_detector.predict(message)
        
        # Get next question
        response_data = self.get_next_question(user_id, emotion, confidence, message)
        
        # Add emotion data
        response_data["emotion"] = emotion
        response_data["confidence"] = confidence
        
        # Occasionally evolve the genetic algorithm
        state = self.get_user_state(user_id)
        state["last_response"] = response_data["reply"]
        
        # Evolve every 10 turns across all users
        total_turns = sum(s["turn_count"] for s in self.user_states.values())
        if total_turns % 10 == 0 and total_turns > 0:
            self.ga.evolve()
            
            # Update all users to use better paths
            for uid, ustate in self.user_states.items():
                # Only update users who haven't progressed far
                if ustate["current_stage_index"] <= 1:
                    ustate["path"] = self.ga.best_path.copy()
                    ustate["path_index"] = self.ga.best_path_index
        
        return response_data
        
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the conversation manager and genetic algorithm"""
        ga_stats = self.ga.get_stats()
        
        # Add additional stats
        stats = {
            "users": len(self.user_states),
            "total_turns": sum(s["turn_count"] for s in self.user_states.values()),
            "average_satisfaction": np.mean([s["satisfaction_score"] for s in self.user_states.values()]) if self.user_states else 0.0,
            "ga_stats": ga_stats
        }
        
        return stats

# ======================================================
# Emotion Detector
# ======================================================

class EmotionDetector:
    """
    Static emotion detection class.
    
    In a production system, this would be replaced with a trained
    machine learning model for emotion detection.
    """
    
    @staticmethod
    def predict(text: str) -> Tuple[str, float]:
        """Predict emotion from text with improved negative emotion detection"""
        text = text.lower()
        
        # Add these FIRST in your keyword checks
        if any(word in text for word in ["not good", "not very good", "depressed", "depression"]):
            return "sadness", 0.8
        
        # Check for explicit negative emotions first
        if any(word in text for word in ["sad", "down", "upset", "miserable", "unhappy"]):
            return "sadness", 0.8
        elif any(word in text for word in ["escape", "get away", "distract", "forget"]):
            # "escape" indicates someone wants to avoid their current mood
            return "sadness", 0.7  # They want to escape sadness
        elif any(word in text for word in ["happy", "good", "great", "joy", "wonderful", "excited", "love"]):
            return "joy", 0.8
        elif any(word in text for word in ["angry", "mad", "frustrated", "annoyed", "irritated"]):
            return "anger", 0.7
        elif any(word in text for word in ["afraid", "scared", "nervous", "anxious", "worried", "terrified"]):
            return "fear", 0.7
        elif any(word in text for word in ["surprised", "amazed", "wow", "unexpected", "shocked"]):
            return "surprise", 0.7
        else:
            return "neutral", 0.6

# ======================================================
# Movie Database
# ======================================================

class MovieDatabase:
    """
    Simple movie database for demonstration purposes.
    
    In a production system, this would be replaced with a real database
    or API calls to a movie recommendation service.
    """
    
    def __init__(self):
        self.movies = self._initialize_movies()
        
    def _initialize_movies(self) -> List[Dict[str, Any]]:
        """Initialize a small set of movies for demonstration"""
        return [
            {
                "title": "The Shawshank Redemption",
                "genre": "drama",
                "description": "A man wrongfully convicted of murder forms a deep friendship with a fellow inmate while attempting to escape prison.",
                "emotions": ["sadness", "neutral"]
            },
            {
                "title": "The Godfather",
                "genre": "drama",
                "description": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son.",
                "emotions": ["anger", "neutral"]
            },
            {
                "title": "The Dark Knight",
                "genre": "action",
                "description": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.",
                "emotions": ["anger", "fear"]
            },
            {
                "title": "Pulp Fiction",
                "genre": "crime",
                "description": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption.",
                "emotions": ["surprise", "anger"]
            },
            {
                "title": "Forrest Gump",
                "genre": "drama",
                "description": "The presidencies of Kennedy and Johnson, the Vietnam War, the Watergate scandal and other historical events unfold from the perspective of an Alabama man with an IQ of 75.",
                "emotions": ["joy", "sadness"]
            },
            {
                "title": "Inception",
                "genre": "sci-fi",
                "description": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.",
                "emotions": ["surprise", "neutral"]
            },
            {
                "title": "The Matrix",
                "genre": "sci-fi",
                "description": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers.",
                "emotions": ["surprise", "fear"]
            },
            {
                "title": "Spirited Away",
                "genre": "animation",
                "description": "During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits, and where humans are changed into beasts.",
                "emotions": ["joy", "surprise"]
            },
            {
                "title": "The Lion King",
                "genre": "animation",
                "description": "Lion prince Simba and his father are targeted by his bitter uncle, who wants to ascend the throne himself.",
                "emotions": ["joy", "sadness"]
            },
            {
                "title": "Toy Story",
                "genre": "animation",
                "description": "A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room.",
                "emotions": ["joy", "fear"]
            },
            {
                "title": "The Shining",
                "genre": "horror",
                "description": "A family heads to an isolated hotel for the winter where a sinister presence influences the father into violence, while his psychic son sees horrific forebodings from both past and future.",
                "emotions": ["fear", "anger"]
            },
            {
                "title": "Get Out",
                "genre": "horror",
                "description": "A young African-American visits his white girlfriend's parents for the weekend, where his simmering unease about their reception of him eventually reaches a boiling point.",
                "emotions": ["fear", "surprise"]
            },
            {
                "title": "La La Land",
                "genre": "musical",
                "description": "While navigating their careers in Los Angeles, a pianist and an actress fall in love while attempting to reconcile their aspirations for the future.",
                "emotions": ["joy", "sadness"]
            },
            {
                "title": "The Grand Budapest Hotel",
                "genre": "comedy",
                "description": "A writer encounters the owner of an aging high-class hotel, who tells him of his early years serving as a lobby boy in the hotel's glorious years under an exceptional concierge.",
                "emotions": ["joy", "surprise"]
            },
            {
                "title": "Parasite",
                "genre": "drama",
                "description": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan.",
                "emotions": ["surprise", "anger"]
            },
            {
                "title": "Interstellar",
                "genre": "sci-fi",
                "description": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
                "emotions": ["fear", "sadness"]
            },
            {
                "title": "Whiplash",
                "genre": "drama",
                "description": "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student's potential.",
                "emotions": ["anger", "fear"]
            },
            {
                "title": "The Social Network",
                "genre": "drama",
                "description": "Harvard student Mark Zuckerberg creates the social networking site that would become known as Facebook, but is later sued by two brothers who claimed he stole their idea, and the co-founder who was later squeezed out of the business.",
                "emotions": ["neutral", "anger"]
            },
            {
                "title": "Coco",
                "genre": "animation",
                "description": "Aspiring musician Miguel, confronted with his family's ancestral ban on music, enters the Land of the Dead to find his great-great-grandfather, a legendary singer.",
                "emotions": ["joy", "sadness"]
            },
            {
                "title": "Mad Max: Fury Road",
                "genre": "action",
                "description": "In a post-apocalyptic wasteland, a woman rebels against a tyrannical ruler in search for her homeland with the aid of a group of female prisoners, a psychotic worshiper, and a drifter named Max.",
                "emotions": ["anger", "fear"]
            }
        ]
        
    def get_movie_by_genre(self, genre: str, exclude_titles: List[str] = None) -> Dict[str, Any]:
        """Get a movie by genre"""
        if exclude_titles is None:
            exclude_titles = []
            
        # Filter by genre and exclude already recommended
        matching = [m for m in self.movies 
                   if m["genre"] == genre.lower() and m["title"] not in exclude_titles]
        
        # If no matches, return any movie
        if not matching:
            matching = [m for m in self.movies if m["title"] not in exclude_titles]
            
        # If still no matches, return any movie
        if not matching:
            matching = self.movies
            
        return random.choice(matching)
        
    def get_movie_by_emotion(self, emotion: str, exclude_titles: List[str] = None) -> Dict[str, Any]:
        """Get a movie by emotion"""
        if exclude_titles is None:
            exclude_titles = []
            
        # Filter by emotion and exclude already recommended
        matching = [m for m in self.movies 
                   if emotion in m["emotions"] and m["title"] not in exclude_titles]
        
        # If no matches, return any movie
        if not matching:
            matching = [m for m in self.movies if m["title"] not in exclude_titles]
            
        # If still no matches, return any movie
        if not matching:
            matching = self.movies
            
        return random.choice(matching)

    def get_movie_by_genre_and_emotion(self, genre: str, emotion: str, exclude_titles: List[str] = None) -> Dict[str, Any]:
        """Get a movie matching both genre and emotion for better personalization"""
        if exclude_titles is None:
            exclude_titles = []
            
        # Try to find movies matching both genre and emotion
        matching = [m for m in self.movies 
                   if m["genre"] == genre.lower() and 
                      emotion in m["emotions"] and 
                      m["title"] not in exclude_titles]
        
        # If no matches with both, fall back to genre only
        if not matching:
            matching = [m for m in self.movies 
                       if m["genre"] == genre.lower() and 
                          m["title"] not in exclude_titles]
        
        # If still no matches, try emotion only
        if not matching:
            matching = [m for m in self.movies 
                       if emotion in m["emotions"] and 
                          m["title"] not in exclude_titles]
        
        # Final fallback to any movie
        if not matching:
            matching = [m for m in self.movies if m["title"] not in exclude_titles]
            
        # If absolutely no options, return any movie
        if not matching:
            matching = self.movies
            
        return random.choice(matching)

# ======================================================
# Main interface for other modules
# ======================================================

# Create a singleton instance for use by other modules
conversation_manager = EmotionDrivenConversationManager()

def process_message(user_id: str, message: str) -> Dict[str, Any]:
    """Process a message from a user"""
    return conversation_manager.process_message(user_id, message)

def get_stats() -> Dict[str, Any]:
    """Get statistics about the conversation manager"""
    return conversation_manager.get_stats()