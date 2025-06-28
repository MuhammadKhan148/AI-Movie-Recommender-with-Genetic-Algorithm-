"""
Performance Metrics Monitor - Runs alongside existing system without modifications
"""

import csv
import time
from datetime import datetime

class MetricsCollector:
    def __init__(self, log_file="metrics_log.csv"):
        self.log_file = log_file
        self.start_time = time.time()
        self.metrics = []
        
        # Initialize log file with headers
        with open(log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if f.tell() == 0:  # File is empty
                writer.writerow(['timestamp', 'user_id', 'conversation_length', 
                               'genres_mentioned', 'emotion_changes', 'recommendations_given',
                               'time_to_recommendation', 'user_satisfaction'])
    
    def log_interaction(self, user_id, conversation_data):
        """Log interaction without affecting the main system"""
        metric = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'conversation_length': conversation_data.get('turn_count', 0),
            'genres_mentioned': len(conversation_data.get('genres', [])),
            'emotion_changes': self._count_emotion_changes(conversation_data),
            'recommendations_given': len(conversation_data.get('recommended_movies', [])),
            'time_to_recommendation': self._calculate_time_to_rec(conversation_data),
            'user_satisfaction': conversation_data.get('satisfaction_score', 0.0)
        }
        
        # Save to CSV
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([metric[key] for key in metric.keys()])
    
    def _count_emotion_changes(self, data):
        # Simple emotion change counter
        return 1 if data.get('detected_emotion') else 0
    
    def _calculate_time_to_rec(self, data):
        # Estimate time to recommendation based on path position
        if 'path' in data and 'current_stage_index' in data:
            try:
                rec_index = data['path'].index('recommendation')
                return rec_index - data['current_stage_index']
            except:
                return 0
        return 0

# To use in your existing backend.py, just add these lines at the top:
"""
metrics_collector = MetricsCollector()

# And after processing each message, add:
if 'user_states' in dir(enhanced_genetic_recommender.conversation_manager):
    user_state = enhanced_genetic_recommender.conversation_manager.get_user_state(request.user_id)
    metrics_collector.log_interaction(request.user_id, user_state)
"""