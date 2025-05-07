"""
Unit Tests for Individual Components
Run with: python tests/test_components.py
"""

import unittest
import sys
sys.path.append('..')
from enhanced_genetic_recommender import EmotionDetector, GeneticAlgorithm

class TestEmotionDetector(unittest.TestCase):
    def setUp(self):
        self.detector = EmotionDetector()
    
    def test_emotion_detection(self):
        test_cases = [
            ("I'm happy", "joy"),
            ("I feel sad", "sadness"),
            ("I'm angry", "anger"),
            ("I feel scared", "fear"),
            ("How surprising", "surprise"),
            ("Normal day", "neutral")
        ]
        
        for text, expected_emotion in test_cases:
            emotion, _ = self.detector.predict(text)
            self.assertEqual(emotion, expected_emotion, f"Failed on: {text}")

class TestGeneticAlgorithm(unittest.TestCase):
    def setUp(self):
        self.ga = GeneticAlgorithm(population_size=3)
    
    def test_path_validity(self):
        # Test that all paths start with greeting
        for path in self.ga.population:
            self.assertEqual(path[0], "greeting")
            self.assertIn("recommendation", path)
    
    def test_fitness_update(self):
        initial_scores = len(self.ga.fitness_scores)
        self.ga.update_fitness(0, 1.0)
        self.assertEqual(len(self.ga.fitness_scores), initial_scores + 1)

if __name__ == '__main__':
    unittest.main()