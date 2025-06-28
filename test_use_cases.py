"""
Test Use Cases for Emotion-Driven Movie Recommender
This file contains test scenarios to verify all functionality works correctly
"""

import requests
import json
import time

class TestScenarios:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def run_all_tests(self):
        """Run all test scenarios"""
        print("=" * 50)
        print("Testing Emotion-Driven Movie Recommender")
        print("=" * 50)
        
        # Test 1: Basic conversation flow
        self.test_basic_conversation()
        
        # Test 2: Emotion detection
        self.test_emotion_detection()
        
        # Test 3: Genre extraction
        self.test_genre_extraction()
        
        # Test 4: Edge cases
        self.test_edge_cases()
        
        # Test 5: Genetic algorithm evolution
        self.test_ga_evolution()
        
        # Test 6: Emotion and genre interaction
        self.test_emotion_and_genre_interaction()
        
        # Test 7: Multi-turn conversation consistency
        self.test_multi_turn_conversation_consistency()
        
    def test_basic_conversation(self):
        """Test the basic conversation flow"""
        print("\nTest 1: Basic Conversation Flow")
        print("-" * 30)
        
        # Reset conversation with new user ID
        user_id = f"test_conversation_{int(time.time())}"
        
        conversation = [
            ("hi", "Should detect greeting stage"),
            ("not very good", "Should detect sadness and progress"),
            ("comedy", "Should detect comedy genre and progress"),
            ("Inception", "Should detect movie mention"),
            ("Fast and Furious", "Should provide recommendation"),
            ("that looks good", "Should proceed to feedback"),
            ("I want something different", "Should proceed to refinement")
        ]
        
        for message, expected in conversation:
            response = self.send_message(message, user_id)
            print(f"User: {message}")
            print(f"Bot: {response['reply']}")
            print(f"Emotion: {response['emotion']} (confidence: {response['confidence']})")
            if 'debug_info' in response:
                if 'stage' in response['debug_info']:
                    print(f"Stage: {response['debug_info']['stage']}")
                if 'genres' in response['debug_info'] and response['debug_info']['genres']:
                    print(f"Detected genres: {response['debug_info']['genres']}")
                if 'preferences' in response['debug_info'] and response['debug_info']['preferences']:
                    print(f"Detected preferences: {response['debug_info']['preferences']}")
            print(f"Expected: {expected}")
            print()
            time.sleep(1)
    
    def test_emotion_detection(self):
        """Test emotion detection with various inputs"""
        print("\nTest 2: Emotion Detection")
        print("-" * 30)
        
        emotions_test = [
            ("I'm so happy today!", "joy"),
            ("I'm feeling really down", "sadness"),
            ("This is terrible", "sadness"),
            ("I'm so angry about this", "anger"),
            ("I'm scared and nervous", "fear"),
            ("Wow, that's amazing!", "surprise"),
            ("I need an escape", "sadness"),
            ("It's a normal day", "neutral")
        ]
        
        for message, expected_emotion in emotions_test:
            # Use fresh user for each emotion test
            user_id = f"test_emotion_{int(time.time() * 1000)}"
            response = self.send_message(message, user_id)
            detected = response['emotion']
            confidence = response['confidence']
            
            status = "✓" if detected == expected_emotion else "✗"
            print(f"{status} '{message}' -> Detected: {detected} (Expected: {expected_emotion})")
            print(f"    Confidence: {confidence}")
            print()
            time.sleep(0.1)  # Short delay to ensure unique timestamps
    
    def test_genre_extraction(self):
        """Test genre extraction"""
        print("\nTest 3: Genre Extraction")
        print("-" * 30)
        
        genre_tests = [
            ("I like action movies", ["action"]),
            ("Comedy and drama are my favorites", ["comedy", "drama"]),
            ("Action", ["action"]),
            ("I enjoy sci-fi and fantasy", ["sci-fi", "fantasy"]),
            ("Horror movies are great", ["horror"])
        ]
        
        for message, expected_genres in genre_tests:
            # Use fresh user for each genre test
            user_id = f"test_genre_{int(time.time() * 1000)}"
            response = self.send_message(message, user_id)
            debug_info = response.get('debug_info', {})
            detected_genres = debug_info.get('genres', []) if debug_info else []
            
            status = "✓" if set(detected_genres) == set(expected_genres) else "✗"
            print(f"{status} '{message}' -> Detected: {detected_genres} (Expected: {expected_genres})")
            print()
            time.sleep(0.1)  # Short delay to ensure unique timestamps
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print("\nTest 4: Edge Cases")
        print("-" * 30)
        
        edge_cases = [
            ("", "Empty message"),
            ("   ", "Whitespace only"),
            ("not good at all", "Multiple negative indicators"),
            ("great and terrible", "Mixed emotions"),
            ("Comedy comedyComedy", "Repeated genre"),
            ("x" * 1500, "Very long message")
        ]
        
        for i, (message, description) in enumerate(edge_cases):
            # Use fresh user for each edge case
            user_id = f"test_edge_{i}_{int(time.time())}"
            response = self.send_message(message, user_id)
            print(f"Test: {description}")
            print(f"Message: '{message[:50]}{'...' if len(message) > 50 else ''}'")
            print(f"Response: {response['reply'][:100]}")
            print(f"Emotion: {response['emotion']}")
            print()
    
    def test_ga_evolution(self):
        """Test genetic algorithm evolution"""
        print("\nTest 5: Genetic Algorithm Evolution")
        print("-" * 30)
        
        # Create multiple users to trigger evolution
        users = []
        for i in range(5):
            users.append(f"user_{i}_{int(time.time())}")
        
        print(f"Created {len(users)} test users")
        
        # Get initial stats
        initial_stats = self.get_stats()
        print(f"Initial generation: {initial_stats['ga_stats']['generation']}")
        print(f"Initial best fitness: {initial_stats['ga_stats']['best_fitness']}")
        
        # Simulate conversations to accumulate enough turns
        for user in users:
            self.simulate_conversation(user)
        
        # Get final stats
        final_stats = self.get_stats()
        print(f"Final generation: {final_stats['ga_stats']['generation']}")
        print(f"Final best fitness: {final_stats['ga_stats']['best_fitness']}")
        print(f"Evolution occurred: {'✓' if final_stats['ga_stats']['generation'] > initial_stats['ga_stats']['generation'] else '✗'}")
    
    def simulate_conversation(self, user_id):
        """Simulate a full conversation for a user"""
        messages = [
            "hi",
            "happy",
            "action",
            "I like Fast and Furious",
            "that sounds great",
            "yes"
        ]
        
        for msg in messages:
            self.send_message(msg, user_id)
            time.sleep(0.5)
    
    def send_message(self, message, user_id):
        """Send a message to the API"""
        url = f"{self.base_url}/chat"
        data = {
            "user_id": user_id,
            "message": message
        }
        
        try:
            response = requests.post(url, json=data)
            return response.json()
        except Exception as e:
            print(f"Error sending message: {e}")
            return {"error": str(e)}
    
    def get_stats(self):
        """Get statistics from the API"""
        url = f"{self.base_url}/stats"
        try:
            response = requests.get(url)
            return response.json()
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"error": str(e)}
    
    def test_emotion_and_genre_interaction(self):
        """Test that emotion is maintained when genre preferences are expressed"""
        print("\nTest 6: Emotion and Genre Interaction")
        print("-" * 30)
        
        emotion_genre_tests = [
            (["I'm feeling really happy", "comedy"], "joy", ["comedy"]),
            (["I'm sad today", "action movies"], "sadness", ["action"]),
            (["I'm so angry", "I like horror"], "anger", ["horror"]),
            (["I'm very anxious", "sci-fi"], "fear", ["sci-fi"]),
        ]
        
        for messages, expected_emotion, expected_genres in emotion_genre_tests:
            # Use fresh user for each test
            user_id = f"test_emotion_genre_{int(time.time() * 1000)}"
            
            # Send first message (emotion)
            response1 = self.send_message(messages[0], user_id)
            emotion1 = response1['emotion']
            
            # Send second message (genre)
            response2 = self.send_message(messages[1], user_id)
            emotion2 = response2['emotion']
            debug_info = response2.get('debug_info', {})
            detected_genres = debug_info.get('genres', []) if debug_info else []
            
            # Check if emotion was maintained and genre was detected
            emotional_status = "✓" if emotion2 == expected_emotion else "✗"
            genre_status = "✓" if set(detected_genres) == set(expected_genres) else "✗"
            
            print(f"Messages: {messages[0]} → {messages[1]}")
            print(f"{emotional_status} Emotion maintained: {emotion1} → {emotion2} (Expected: {expected_emotion})")
            print(f"{genre_status} Genres detected: {detected_genres} (Expected: {expected_genres})")
            print()
            time.sleep(1)
    
    def test_multi_turn_conversation_consistency(self):
        """Test longer conversations for state consistency"""
        print("\nTest 7: Multi-turn Conversation Consistency")
        print("-" * 30)
        
        user_id = f"test_long_conversation_{int(time.time())}"
        
        # Simulate a more complex conversation with state checks
        messages = [
            ("hello there", "greeting", None, []),
            ("I'm feeling down today", "emotion_elicitation", "sadness", []),
            ("I like comedy movies", "genre_preference", "sadness", ["comedy"]),
            ("Christopher Nolan is great", "specific_preference", "sadness", ["comedy"]),
            ("that sounds good", "feedback", "sadness", ["comedy"]),
            ("something different please", "refinement", "sadness", ["comedy"])
        ]
        
        state_history = []
        
        for msg, expected_stage, expected_emotion, expected_genres in messages:
            response = self.send_message(msg, user_id)
            debug_info = response.get('debug_info', {})
            
            current_state = {
                "message": msg,
                "stage": debug_info.get('stage', 'unknown'),
                "emotion": response['emotion'],
                "genres": debug_info.get('genres', []) if debug_info else []
            }
            
            state_history.append(current_state)
            
            # Check expectations
            stage_ok = current_state["stage"] == expected_stage if expected_stage else True
            emotion_ok = current_state["emotion"] == expected_emotion if expected_emotion else True
            genres_ok = set(current_state["genres"]) == set(expected_genres) if expected_genres else True
            
            status = "✓" if stage_ok and emotion_ok and genres_ok else "✗"
            print(f"{status} Message: {msg}")
            print(f"  Stage: {current_state['stage']} (Expected: {expected_stage})")
            print(f"  Emotion: {current_state['emotion']} (Expected: {expected_emotion})")
            print(f"  Genres: {current_state['genres']} (Expected: {expected_genres})")
            print()
            
            time.sleep(1)
        
        # At the end, print a summary of the conversation state changes
        print("Conversation State Evolution:")
        for i, state in enumerate(state_history):
            print(f"Turn {i+1}: {state['message']}")
            print(f"  → Stage: {state['stage']}, Emotion: {state['emotion']}, Genres: {state['genres']}")

if __name__ == "__main__":
    # Run all tests
    tester = TestScenarios()
    tester.run_all_tests()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)