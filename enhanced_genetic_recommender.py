"""
Enhanced Emotion-Driven Movie Recommender with MovieLens Data Integration
FINAL FIX - All conversation flow and genre extraction issues resolved
"""

import random
import time
import re
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

# Disable ML emotion detector for testing - use simple keyword-based instead
USE_ML_EMOTIONS = False

# ======================================================
# Genetic Algorithm
# ======================================================
class GeneticAlgorithm:
    def __init__(self, population_size: int = 10, mutation_rate: float = 0.2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation = 0
        
        self.available_stages = [
            "greeting",
            "emotion_elicitation",
            "genre_preference",
            "specific_preference",
            "recommendation",
            "feedback",
            "refinement"
        ]
        
        self.population = self._initialize_population()
        self.fitness_scores = {}
        self.best_path_index = 0
        self.best_path = self.population[0]
        self.average_fitness = 0.0
        self.best_fitness = 0.0
        
    def _initialize_population(self) -> List[List[str]]:
        population = []
        
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
        
        for _ in range(self.population_size - 1):
            path = ["greeting"]
            intermediate_stages = random.sample(
                self.available_stages[1:-1],
                random.randint(2, 5)
            )
            path.extend(intermediate_stages)
            
            if "recommendation" not in path:
                path.append("recommendation")
                
            if path[-1] != "refinement":
                path.append("refinement")
                
            population.append(path)
            
        return population
    
    def select_parents(self) -> List[List[str]]:
        if not self.fitness_scores:
            return random.sample(self.population, 2)
        
        sorted_paths = sorted(
            self.fitness_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        best_indices = [idx for idx, _ in sorted_paths[:2]]
        return [self.population[idx] for idx in best_indices]
    
    def crossover(self, parents: List[List[str]]) -> List[List[str]]:
        if len(parents) < 2:
            return parents
            
        parent1, parent2 = parents
        min_len = min(len(parent1), len(parent2))
        
        if min_len <= 2:
            return parents
            
        crossover_point = random.randint(1, min_len - 1)
        
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        
        for child in [child1, child2]:
            if "recommendation" not in child:
                position = random.randint(1, len(child) - 1)
                child.insert(position, "recommendation")
                
            if child[-1] != "refinement":
                child.append("refinement")
                
        return [child1, child2]
    
    def mutate(self, offspring: List[List[str]]) -> List[List[str]]:
        mutated = []
        
        for path in offspring:
            mutated_path = path.copy()
            
            if random.random() < self.mutation_rate:
                if len(mutated_path) > 2:
                    position = random.randint(1, len(mutated_path) - 2)
                    current = mutated_path[position]
                    options = [s for s in self.available_stages 
                              if s != current and s != "greeting"]
                    mutated_path[position] = random.choice(options)
            
            mutated.append(mutated_path)
            
        return mutated
    
    def evolve(self) -> None:
        if not self.fitness_scores:
            return
            
        parents = self.select_parents()
        offspring = self.crossover(parents)
        mutated_offspring = self.mutate(offspring)
        
        if self.fitness_scores:
            sorted_indices = [idx for idx, _ in 
                             sorted(self.fitness_scores.items(), 
                                    key=lambda x: x[1])]
            
            for i, idx in enumerate(sorted_indices):
                if i < len(mutated_offspring):
                    self.population[idx] = mutated_offspring[i]
                    
        self.generation += 1
        
        if self.fitness_scores:
            best_idx = max(self.fitness_scores.items(), key=lambda x: x[1])[0]
            self.best_path_index = best_idx
            self.best_path = self.population[best_idx]
            self.best_fitness = self.fitness_scores[best_idx]
            self.average_fitness = sum(self.fitness_scores.values()) / len(self.fitness_scores)
    
    def update_fitness(self, path_index: int, reward: float) -> None:
        if path_index not in self.fitness_scores:
            self.fitness_scores[path_index] = 0
            
        self.fitness_scores[path_index] += reward
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "best_path": self.best_path,
            "best_fitness": self.best_fitness,
            "average_fitness": self.average_fitness,
            "mutation_rate": self.mutation_rate,
            "population": self.population[:3],
            "fitness_scores": dict(sorted(self.fitness_scores.items())[:5]) if self.fitness_scores else {}
        }

# ======================================================
# Enhanced Movie Database using MovieLens
# ======================================================
class EnhancedMovieDB:
    def __init__(self, data_path="data/raw/ml-20m"):
        try:
            self.movies = pd.read_csv(f"{data_path}/movies.csv")
            self.ratings = pd.read_csv(f"{data_path}/ratings.csv")
            self.enabled = True
            print(f"Loaded {len(self.movies)} movies from MovieLens dataset")
        except:
            print("MovieLens data not found, using simplified database")
            self.enabled = False
            self.movies = self._get_simple_movies()
    
    def _get_simple_movies(self):
        # Fallback to simple movie list if MovieLens data not available
        return pd.DataFrame([
            {"movieId": 1, "title": "Inception (2010)", "genres": "Sci-Fi|Thriller"},
            {"movieId": 2, "title": "Toy Story (1995)", "genres": "Animation|Children|Comedy"},
            {"movieId": 3, "title": "The Dark Knight (2008)", "genres": "Action|Crime|Drama"},
            {"movieId": 4, "title": "La La Land (2016)", "genres": "Musical|Romance"}
        ])
    
    def get_movie_by_genre(self, genre: str, exclude_titles: List[str] = None) -> Dict[str, Any]:
        if exclude_titles is None:
            exclude_titles = []
        
        # Filter movies by genre
        matching = self.movies[self.movies["genres"].str.contains(genre, case=False, na=False)]
        available = matching[~matching["title"].isin(exclude_titles)]
        
        if len(available) > 0:
            selected = available.sample(1).iloc[0]
            # Extract year if present in title
            year_match = re.search(r'\((\d{4})\)', selected["title"])
            year = year_match.group(1) if year_match else "Unknown"
            
            # Extract main genre
            genre_list = selected["genres"].split("|")
            main_genre = genre_list[0] if genre_list else genre
            
            return {
                "title": selected["title"],
                "genre": main_genre,
                "description": f"A {main_genre} movie from {year}."
            }
        
        # Fallback if no matches
        return {
            "title": "Unknown Movie",
            "genre": genre,
            "description": "Movie description not available."
        }
    
    def get_movie_by_emotion(self, emotion: str, exclude_titles: List[str] = None) -> Dict[str, Any]:
        # Map emotions to genres
        emotion_to_genre = {
            'joy': 'Comedy',
            'sadness': 'Comedy',  # When sad, recommend comedy to cheer up
            'anger': 'Action',
            'fear': 'Comedy',     # When fearful, recommend comedy to distract
            'surprise': 'Thriller',
            'neutral': 'Drama'
        }
        genre = emotion_to_genre.get(emotion, 'Drama')
        return self.get_movie_by_genre(genre, exclude_titles)

# ======================================================
# Emotion Detector (simple keyword-based)
# ======================================================
class EmotionDetector:
    @staticmethod
    def predict(text: str) -> Tuple[str, float]:
        text = text.lower()
        
        # Check for explicit negative emotions FIRST
        if any(word in text for word in ["not good", "not very good", "depressed", "depression", "terrible", "awful", "horrible"]):
            return "sadness", 0.8
        elif any(word in text for word in ["feeling down", "feeling really down", "feeling bad", "feel bad", "down"]):
            return "sadness", 0.8
        elif any(word in text for word in ["sad", "upset", "miserable", "unhappy"]):
            return "sadness", 0.7
        elif any(word in text for word in ["escape", "distract", "need a break"]):
            # When someone wants to escape, they're usually unhappy
            return "sadness", 0.7
        elif any(word in text for word in ["happy", "good", "great", "joy", "wonderful", "excited", "love"]):
            return "joy", 0.8
        elif any(word in text for word in ["angry", "mad", "frustrated", "annoyed", "irritated"]):
            return "anger", 0.7
        elif any(word in text for word in ["afraid", "scared", "nervous", "anxious", "worried", "terrified"]):
            return "fear", 0.7
        elif any(word in text for word in ["surprised", "surprising", "amazed", "wow", "unexpected", "shocking", "how surprising"]):
            return "surprise", 0.7
        elif any(word in text for word in ["normal", "okay", "alright", "fine"]):
            return "neutral", 0.6
        else:
            return "neutral", 0.5

# ======================================================
# Enhanced Conversation Manager
# ======================================================
class EnhancedConversationManager:
    def __init__(self):
        # Use simple keyword emotion detector for testing
        self.emotion_detector = EmotionDetector()
        
        self.ga = GeneticAlgorithm()
        self.movie_db = EnhancedMovieDB()
        self.user_states = {}
        
        self.questions = self._initialize_questions()
        self.emotion_genre_mapping = self._initialize_emotion_mapping()
    
    def _initialize_questions(self) -> Dict[str, List[str]]:
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
        return {
            'joy': {
                'primary': ['Comedy', 'Animation', 'Musical'],
                'secondary': ['Adventure', 'Family', 'Fantasy']
            },
            'sadness': {
                'primary': ['Comedy', 'Animation'],  # When sad, recommend uplifting genres
                'secondary': ['Adventure', 'Musical', 'Family']
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
        if user_id not in self.user_states:
            # Initialize new user
            self.user_states[user_id] = {
                "current_stage_index": 0,
                "path": self.ga.best_path.copy(),
                "path_index": self.ga.best_path_index,
                "detected_emotion": None,
                "emotion_confidence": 0,
                "genres": [],
                "specific_preferences": [],
                "recommended_movies": [],
                "consecutive_yes_count": 0,
                "satisfaction_score": 0.0,
                "turn_count": 0,
                "last_response": ""
            }
            # INITIALIZE FITNESS FOR THIS PATH
            self.ga.update_fitness(self.ga.best_path_index, 0.0)
            
        return self.user_states[user_id]
    
    def _extract_genres(self, message: str) -> List[str]:
        # FIXED: Improved genre extraction
        genres = []
        message_lower = message.lower()
        common_genres = [
            "action", "adventure", "comedy", "drama", "horror", 
            "thriller", "sci-fi", "fantasy", "romance", "documentary",
            "animation", "mystery", "crime", "musical", "western",
            "superhero", "history", "war"
        ]
        
        # Check for capitalized genre names
        message_words = message.split()
        for word in message_words:
            if word.title() in ["Action", "Comedy", "Drama", "Horror", "Thriller", "Romance", "Animation", "Mystery"]:
                if word.lower() not in genres:
                    genres.append(word.lower())
        
        # Check for lowercase genres in the message
        for genre in common_genres:
            if genre in message_lower:
                if genre not in genres:
                    genres.append(genre)
        
        # Handle special cases like "sci-fi"
        if "sci fi" in message_lower or "science fiction" in message_lower:
            if "sci-fi" not in genres:
                genres.append("sci-fi")
        
        return genres
    
    def _extract_movie_mentions(self, message: str) -> List[str]:
        quoted = re.findall(r'"([^"]*)"', message)
        quoted.extend(re.findall(r"'([^']*)'", message))
        
        movie_patterns = [
            r"(?:watched|liked|loved|enjoyed|recommended|seen)\s+([A-Z][a-zA-Z0-9\s]+)",
            r"(?:favorite movie is|best movie|great movie)\s+([A-Z][a-zA-Z0-9\s]+)"
        ]
        
        for pattern in movie_patterns:
            matches = re.findall(pattern, message)
            quoted.extend(matches)
        
        # Add simple capitalized words as movie titles
        for word in message.split():
            if word[0].isupper() and len(word) > 3 and not word.isupper():
                if word not in quoted:
                    quoted.append(word)
        
        return list(set(quoted))
    
    def process_message(self, user_id: str, message: str) -> Dict[str, Any]:
        if not message or not message.strip():
            return {
                "reply": "I didn't catch that. Could you please say something?",
                "emotion": "neutral",
                "confidence": 0.5,
                "movies": [],
                "debug_info": {"error": "empty_message"}
            }
        
        if len(message) > 1000:
            message = message[:1000] + "..."
            
        # Get emotion from message
        emotion, confidence = self.emotion_detector.predict(message)
        
        # Get user state
        state = self.get_user_state(user_id)
        
        # Update emotion ONLY if:
        # 1. We detect a strong emotion (confidence > 0.4)
        # AND
        # 2. Either we have no emotion yet OR the new emotion has higher confidence
        if confidence > 0.4 and (state["detected_emotion"] is None or confidence > state["emotion_confidence"]):
            state["detected_emotion"] = emotion
            state["emotion_confidence"] = confidence
        
        # Get next question
        response_data = self.get_next_question(user_id, emotion, confidence, message)
        
        # Add user's persistent emotion data, not just the newly detected emotion
        response_data["emotion"] = state["detected_emotion"] or emotion
        response_data["confidence"] = state["emotion_confidence"] or confidence
        
        # Evolve genetic algorithm occasionally
        state = self.get_user_state(user_id)
        state["last_response"] = response_data["reply"]
        
        total_turns = sum(s["turn_count"] for s in self.user_states.values())
        if total_turns % 10 == 0 and total_turns > 0:
            self.ga.evolve()
            
            # Update all users to use better paths
            for uid, ustate in self.user_states.items():
                if ustate["current_stage_index"] <= 1:
                    ustate["path"] = self.ga.best_path.copy()
                    ustate["path_index"] = self.ga.best_path_index
        
        return response_data
    
    def get_next_question(self, user_id: str, emotion: str, confidence: float, message: str) -> Dict[str, Any]:
        state = self.get_user_state(user_id)
        state["turn_count"] += 1
        
        # Print debug info for testing
        print(f"User {user_id}, Stage index: {state['current_stage_index']}, Path length: {len(state['path'])}")
        
        # Extract current stage from path
        if state["current_stage_index"] < len(state["path"]):
            current_stage = state["path"][state["current_stage_index"]]
        else:
            current_stage = "recommendation"
        
        # Extract genres if mentioned
        extracted_genres = self._extract_genres(message)
        print(f"Extracted genres from '{message}': {extracted_genres}")  # Debug print
        
        if extracted_genres:
            state["genres"].extend([g for g in extracted_genres if g not in state["genres"]])
            self.ga.update_fitness(state["path_index"], 1.0)
            print(f"State genres after update: {state['genres']}")  # Debug print
        
        # Check for explicit movie mentions for specific preferences
        movie_mentions = self._extract_movie_mentions(message)
        if movie_mentions:
            state["specific_preferences"].extend([m for m in movie_mentions if m not in state["specific_preferences"]])
            self.ga.update_fitness(state["path_index"], 0.5)
        
        # Process stage-specific logic
        if current_stage == "greeting":
            response = random.choice(self.questions["greeting"])
            state["current_stage_index"] += 1
            
            return {
                "reply": response,
                "movies": [],
                "debug_info": {"stage": current_stage, "path": state["path"], "genres": state["genres"]}
            }
            
        elif current_stage == "emotion_elicitation":
            # Update emotion if detected with high confidence and either:
            # 1. We don't have an emotion yet OR
            # 2. New emotion has higher confidence than existing one
            if confidence > 0.4 and (state["detected_emotion"] is None or confidence > state["emotion_confidence"]):
                state["detected_emotion"] = emotion
                state["emotion_confidence"] = confidence
            
            response = random.choice(self.questions["emotion_elicitation"])
            
            # FIXED: Progress only when emotion is detected or if user gives a proper response
            if state["detected_emotion"] and confidence > 0.4:
                state["current_stage_index"] += 1
            
            return {
                "reply": response,
                "movies": [],
                "debug_info": {"stage": current_stage, "path": state["path"], "emotion": state["detected_emotion"], "genres": state["genres"]}
            }
            
        elif current_stage == "genre_preference":
            # Skip if we already have genres
            if state["genres"]:
                state["current_stage_index"] += 1
                # Skip to next stage without asking again
                return self.get_next_question(user_id, emotion, confidence, message)
            
            response = random.choice(self.questions["genre_preference"])
            
            # Progress if we have genres
            if state["genres"]:
                state["current_stage_index"] += 1
            
            return {
                "reply": response,
                "movies": [],
                "debug_info": {"stage": current_stage, "path": state["path"], "genres": state["genres"]}
            }
            
        elif current_stage == "specific_preference":
            response = random.choice(self.questions["specific_preference"])
            state["current_stage_index"] += 1
            
            return {
                "reply": response,
                "movies": [],
                "debug_info": {"stage": current_stage, "path": state["path"], "preferences": state["specific_preferences"], "genres": state["genres"]}
            }
            
        elif current_stage == "recommendation":
            # Generate movie recommendation from MovieLens data
            if state["genres"]:
                # Use the last mentioned genre
                genre = state["genres"][-1]
                movie = self.movie_db.get_movie_by_genre(genre, state["recommended_movies"])
            else:
                emotion = state["detected_emotion"] or "neutral"
                movie = self.movie_db.get_movie_by_emotion(emotion, state["recommended_movies"])
            
            state["recommended_movies"].append(movie["title"])
            
            # Fill in template
            template = random.choice(self.questions["recommendation"])
            emotion_adj = self._get_emotion_adjective(state["detected_emotion"] or "neutral")
            
            # Use actual detected emotion
            detected_emotion = state["detected_emotion"] or "current"
            
            response = template.replace("[MOVIE]", movie["title"])
            response = response.replace("[GENRE]", movie["genre"])
            response = response.replace("[EMOTION]", detected_emotion)
            response = response.replace("[FEATURE]", emotion_adj + " storytelling")
            response = response.replace("[QUALITY]", emotion_adj + " elements")
            
            state["current_stage_index"] += 1
            
            return {
                "reply": response,
                "movies": [{
                    "title": movie["title"], 
                    "genre": movie["genre"],
                    "description": movie["description"],
                    "score": 0.95
                }],
                "debug_info": {"stage": current_stage, "path": state["path"], "recommendation": movie["title"], "genres": state["genres"]}
            }
            
        elif current_stage == "feedback":
            response = random.choice(self.questions["feedback"])
            state["current_stage_index"] += 1
            
            # Check for positive feedback
            if any(word in message.lower() for word in ["good", "great", "nice", "perfect", "love", "thanks", "yes", "okay"]):
                self.ga.update_fitness(state["path_index"], 2.0)
                state["satisfaction_score"] += 1.0
            elif any(word in message.lower() for word in ["bad", "terrible", "not", "don't", "didn't", "no"]):
                self.ga.update_fitness(state["path_index"], -1.0)
                state["satisfaction_score"] -= 0.5
                
            return {
                "reply": response,
                "movies": [],
                "debug_info": {"stage": current_stage, "path": state["path"], "genres": state["genres"]}
            }
            
        elif current_stage == "refinement":
            response = random.choice(self.questions["refinement"])
            
            try:
                recommend_idx = state["path"].index("recommendation")
                state["current_stage_index"] = recommend_idx
            except ValueError:
                state["current_stage_index"] = 0
                
            return {
                "reply": response,
                "movies": [],
                "debug_info": {"stage": current_stage, "path": state["path"], "genres": state["genres"]}
            }
    
    def _get_emotion_adjective(self, emotion: str) -> str:
        emotion_map = {
            "joy": "uplifting",
            "sadness": "comforting",  # Changed from "reflective" to comfort sad users
            "anger": "cathartic",
            "fear": "comforting",
            "surprise": "thought-provoking",
            "neutral": "engaging"
        }
        return emotion_map.get(emotion, "enjoyable")
    
    def get_stats(self) -> Dict[str, Any]:
        ga_stats = self.ga.get_stats()
        
        stats = {
            "users": len(self.user_states),
            "total_turns": sum(s["turn_count"] for s in self.user_states.values()),
            "average_satisfaction": np.mean([s["satisfaction_score"] for s in self.user_states.values()]) if self.user_states else 0.0,
            "ga_stats": ga_stats
        }
        
        return stats

# Create singleton instance
conversation_manager = EnhancedConversationManager()

# Main interface functions
def process_message(user_id: str, message: str) -> Dict[str, Any]:
    return conversation_manager.process_message(user_id, message)

def get_stats() -> Dict[str, Any]:
    return conversation_manager.get_stats()