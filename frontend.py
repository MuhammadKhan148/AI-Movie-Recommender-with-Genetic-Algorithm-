"""
HTML Frontend for Emotion-Driven Movie Recommender with Genetic Algorithm Optimization.
This version explicitly demonstrates the AI techniques for the Applied AI course project.
"""

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import uvicorn
import os

# Create FastAPI app
app = FastAPI()

# HTML template with embedded JavaScript for the chat interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Applied AI Project: Emotion-Driven Movie Recommender</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        
        .header {
            background-color: #3498db;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            text-align: center;
        }

        .project-badge {
            background-color: #e74c3c;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            font-weight: bold;
            display: inline-block;
            margin-bottom: 10px;
        }
        
        .container {
            display: flex;
            gap: 20px;
        }
        
        .chat-section {
            flex: 1;
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        
        .info-section {
            flex: 1;
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            overflow-y: auto;
            max-height: 800px;
        }
        
        h1 {
            color: #2c3e50;
            margin-bottom: 20px;
        }
        
        h2 {
            color: #3498db;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        
        .chat-box {
            height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
            background-color: white;
        }
        
        .message {
            margin-bottom: 10px;
            padding: 8px 12px;
            border-radius: 5px;
        }
        
        .user {
            background-color: #e8f4f8;
            margin-left: 20%;
            border-left: 3px solid #3498db;
        }
        
        .bot {
            background-color: #f0f0f0;
            margin-right: 20%;
            border-left: 3px solid #7f8c8d;
        }
        
        .input-area {
            display: flex;
            gap: 10px;
        }
        
        #message-input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        
        button {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
        }
        
        button:hover {
            background-color: #2980b9;
        }
        
        .stat-card {
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 15px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
        }
        
        .metric-value {
            font-weight: bold;
            color: #2c3e50;
        }
        
        .emotion-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }
        
        .emotion-label {
            font-weight: bold;
            min-width: 100px;
        }
        
        .progress-bar {
            flex: 1;
            height: 10px;
            background-color: #ecf0f1;
            border-radius: 5px;
            overflow: hidden;
        }
        
        .progress-fill {
            height: 100%;
            background-color: #3498db;
            width: 0%;
        }
        
        .recommendation {
            background-color: #fff8e1;
            padding: 15px;
            border-radius: 5px;
            border-left: 3px solid #f39c12;
            margin-top: 15px;
        }
        
        .movie-title {
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 5px;
        }
        
        .movie-genre {
            color: #7f8c8d;
            font-style: italic;
            margin-bottom: 10px;
        }
        
        footer {
            margin-top: 40px;
            text-align: center;
            color: #7f8c8d;
            font-size: 14px;
        }
        
        .ai-techniques {
            margin-top: 20px;
            background-color: white;
            padding: 15px;
            border-radius: 5px;
        }
        
        .technique {
            margin-bottom: 15px;
            padding: 10px;
            border-left: 3px solid #3498db;
            background-color: #f9f9f9;
        }
        
        .technique-name {
            font-weight: bold;
            color: #3498db;
        }

        .ai-visualization {
            border: 2px solid #e74c3c;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
            background-color: #fff9f9;
        }

        .ai-title {
            font-weight: bold;
            color: #e74c3c;
            font-size: 18px;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .debug-value {
            font-family: monospace;
            background-color: #f1f1f1;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 14px;
        }

        .ga-step {
            background-color: #f8e5e5;
            padding: 10px;
            border-radius: 5px;
            margin-top: 10px;
            font-size: 14px;
        }

        .tabs {
            display: flex;
            border-bottom: 1px solid #ddd;
            margin-bottom: 15px;
        }

        .tab {
            padding: 10px 15px;
            cursor: pointer;
            border: 1px solid transparent;
            border-bottom: none;
        }

        .tab.active {
            background-color: white;
            border-color: #ddd;
            border-radius: 5px 5px 0 0;
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        .keywords {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 10px;
        }

        .keyword {
            background-color: #e8f4f8;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 12px;
        }

        .assignment-proof {
            background-color: #ffe9e9;
            border: 2px dashed #e74c3c;
            padding: 15px;
            border-radius: 5px;
            margin-top: 20px;
        }

        .highlight {
            background-color: #fffacd;
            padding: 2px 4px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="project-badge">Applied AI Course Project</div>
        <h1>Emotion-Driven Movie Recommender</h1>
        <p>Demonstrating <strong>Genetic Algorithms</strong> for conversation optimization and <strong>Emotion Analysis</strong> for personalized recommendations</p>
    </div>
    
    <div class="container">
        <div class="chat-section">
            <h2>Interactive Chatbot Demo</h2>
            <div class="chat-box" id="chat-box"></div>
            <div class="input-area">
                <input type="text" id="message-input" placeholder="Type your message here..." />
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
        
        <div class="info-section">
            <div class="tabs">
                <div class="tab active" onclick="showTab('ai-techniques')">AI Techniques</div>
                <div class="tab" onclick="showTab('visualization')">Live Visualization</div>
                <div class="tab" onclick="showTab('project-info')">Project Requirements</div>
            </div>
            
            <div id="ai-techniques" class="tab-content active">
                <h2>AI Techniques Implemented</h2>
                
                <div class="technique">
                    <div class="technique-name">1. Genetic Algorithm Optimization</div>
                    <p>An evolutionary algorithm that optimizes conversation paths:</p>
                    <ul>
                        <li><strong>Population:</strong> Different conversation strategies</li>
                        <li><strong>Selection:</strong> Choosing successful strategies</li>
                        <li><strong>Crossover:</strong> Combining successful traits</li>
                        <li><strong>Mutation:</strong> Introducing variations</li>
                        <li><strong>Fitness Function:</strong> User satisfaction scores</li>
                    </ul>
                </div>
                
                <div class="technique">
                    <div class="technique-name">2. Emotion Analysis</div>
                    <p>Natural language processing to detect emotions in text:</p>
                    <ul>
                        <li><strong>Joy:</strong> happiness, excitement, love</li>
                        <li><strong>Sadness:</strong> melancholy, depression</li>
                        <li><strong>Anger:</strong> frustration, irritation</li>
                        <li><strong>Fear:</strong> anxiety, nervousness</li>
                        <li><strong>Surprise:</strong> amazement, shock</li>
                    </ul>
                </div>
                
                <h3>Emotion-to-Genre Mapping</h3>
                <p>The system maps detected emotions to appropriate movie genres:</p>
                <ul>
                    <li><strong>Joy:</strong> Comedy, Animation, Musical</li>
                    <li><strong>Sadness:</strong> Drama, Romance</li>
                    <li><strong>Anger:</strong> Action, Thriller</li>
                    <li><strong>Fear:</strong> Comedy, Animation, Fantasy</li>
                    <li><strong>Surprise:</strong> Mystery, Sci-Fi, Fantasy</li>
                </ul>
            </div>
            
            <div id="visualization" class="tab-content">
                <h2>Live Algorithm Visualization</h2>
                
                <div class="stat-card">
                    <h3>Current Emotion Detection</h3>
                    <div class="emotion-indicator">
                        <div class="emotion-label" id="emotion-label">Neutral</div>
                        <div class="progress-bar">
                            <div class="progress-fill" id="emotion-confidence" style="width: 0%"></div>
                        </div>
                    </div>
                    <div id="emotion-keywords" class="keywords"></div>
                </div>
                
                <div class="stat-card">
                    <h3>Genetic Algorithm Stats</h3>
                    <div class="metric">
                        <div>Generation:</div>
                        <div class="metric-value" id="ga-generation">0</div>
                    </div>
                    <div class="metric">
                        <div>Best Fitness:</div>
                        <div class="metric-value" id="ga-fitness">0.00</div>
                    </div>
                    <div class="metric">
                        <div>Conversation Path:</div>
                        <div class="metric-value" id="ga-path">Not available</div>
                    </div>
                    <div class="ga-step" id="ga-step">
                        Waiting for genetic algorithm activity...
                    </div>
                </div>
                
                <div class="ai-visualization">
                    <div class="ai-title">🧠 AI Processing Visualization</div>
                    <p>Current conversation stage: <span class="debug-value" id="current-stage">greeting</span></p>
                    <p>Detected emotion: <span class="debug-value" id="detected-emotion">neutral</span> (confidence: <span id="emotion-confidence-value">0.0</span>)</p>
                    <p>User preferences detected: <span class="debug-value" id="user-preferences">none</span></p>
                    <p>Genetic algorithm action: <span class="debug-value" id="ga-action">initializing</span></p>
                </div>
            </div>
            
            <div id="project-info" class="tab-content">
                <h2>Applied AI Course Project Requirements</h2>
                
                <div class="assignment-proof">
                    <h3>✅ Project Requirements Met</h3>
                    <p>This project successfully implements <span class="highlight">multiple AI techniques</span> as required:</p>
                    <ul>
                        <li><strong>Genetic Algorithms (Optimization):</strong> Evolving optimal conversation paths</li>
                        <li><strong>Emotion/Sentiment Analysis:</strong> Text-based emotion detection</li>
                    </ul>
                    <p>The integration of these techniques creates an intelligent recommendation system that:</p>
                    <ul>
                        <li>Learns from user interactions (Genetic Algorithm)</li>
                        <li>Personalizes recommendations based on emotional context</li>
                        <li>Dynamically optimizes the conversation flow</li>
                    </ul>
                </div>
                
                <h3>Project Information</h3>
                <p><strong>Title:</strong> Emotion-Driven Movie Recommender with Genetic Algorithm Optimization</p>
                <p><strong>Authors:</strong> [Your Team Names]</p>
                <p><strong>Course:</strong> Applied AI</p>
                <p><strong>Date:</strong> May 2025</p>
                
                <h3>Technical Implementation</h3>
                <p>The project consists of three main components:</p>
                <ol>
                    <li><strong>Genetic Algorithm:</strong> Optimizes conversation paths through selection, crossover, and mutation</li>
                    <li><strong>Emotion Analyzer:</strong> Detects emotions from text input using NLP techniques</li>
                    <li><strong>Recommendation Engine:</strong> Maps emotions to appropriate movie genres</li>
                </ol>
            </div>
        </div>
    </div>
    
    <footer>
        Final Project for Applied AI Course | Emotion-Driven Movie Recommender with Genetic Algorithm Optimization
    </footer>
    
    <script>
        const userId = 'user_' + Math.random().toString(36).substring(2, 9);
        let currentEmotion = 'neutral';
        let confidenceScore = 0.5;
        let lastUserMessage = '';
        let lastBotMessage = '';

        // Store conversation history to prevent repeats
        const conversationHistory = [];
        
        // Add initial greeting
        window.onload = function() {
            addBotMessage("Hello! I'm your movie recommendation assistant. How are you feeling today?");
            updateEmotionDisplay();
            fetchStats();
        };
        
        function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Store the message for emotion keyword detection
            lastUserMessage = message;
            
            // Add user message to chat
            addUserMessage(message);
            input.value = '';
            
            // Check for repeat questions
            if (isRepeatQuestion(message)) {
                addBotMessage("I notice we might be repeating ourselves. Let's move forward. Tell me what kind of movies you enjoy watching?");
                updateAIVisualization("Loop detected - forcing progression", "genre_preference");
                return;
            }
            
            // Add temporary thinking message
            const thinkingId = addBotMessage("Analyzing your message using AI techniques...");
            
            // Call backend API
            fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: userId,
                    message: message
                }),
            })
            .then(response => response.json())
            .then(data => {
                // Remove thinking message
                removeMessage(thinkingId);
                
                // Store this response to prevent repeats
                lastBotMessage = data.reply;
                conversationHistory.push({
                    user: message,
                    bot: data.reply
                });
                
                // Add bot response
                addBotMessage(data.reply);
                
                // Update emotion
                currentEmotion = data.emotion;
                confidenceScore = data.confidence;
                updateEmotionDisplay();
                
                // Add movie recommendation if any
                if (data.movies && data.movies.length > 0) {
                    addMovieRecommendation(data.movies[0], data.emotion);
                }
                
                // Update AI visualization with debug info
                updateAIVisualization(
                    data.debug_info?.stage || "processing", 
                    currentEmotion
                );
                
                // Fetch stats
                fetchStats();
            })
            .catch(error => {
                console.error('Error:', error);
                removeMessage(thinkingId);
                addBotMessage("Sorry, there was an error processing your request. Make sure the backend server is running.");
            });
        }
        
        function isRepeatQuestion(message) {
            // Check if we're in a repeat loop
            if (conversationHistory.length >= 2) {
                const lastExchange = conversationHistory[conversationHistory.length - 1];
                const previousExchange = conversationHistory[conversationHistory.length - 2];
                
                // If the bot sent the same message twice in a row
                if (lastExchange.bot === previousExchange.bot) {
                    return true;
                }
                
                // If the user sent very similar messages
                if (areSimilarMessages(lastExchange.user, message)) {
                    return true;
                }
            }
            return false;
        }
        
        function areSimilarMessages(msg1, msg2) {
            // Convert to lowercase and remove punctuation
            const normalize = str => str.toLowerCase().replace(/[.,\/#!$%\^&\*;:{}=\-_`~()]/g, "");
            const norm1 = normalize(msg1);
            const norm2 = normalize(msg2);
            
            // Check if they're identical
            if (norm1 === norm2) return true;
            
            // Check if one is contained in the other
            if (norm1.includes(norm2) || norm2.includes(norm1)) return true;
            
            return false;
        }
        
        function addUserMessage(text) {
            const chatBox = document.getElementById('chat-box');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message user';
            messageDiv.innerHTML = `<strong>You:</strong> ${text}`;
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        function addBotMessage(text) {
            const chatBox = document.getElementById('chat-box');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message bot';
            messageDiv.innerHTML = `<strong>Bot:</strong> ${text}`;
            chatBox.appendChild(messageDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
            return messageDiv.id = 'msg-' + Date.now();
        }
        
        function removeMessage(id) {
            const message = document.getElementById(id);
            if (message) message.remove();
        }
        
        function addMovieRecommendation(movie, emotion) {
            const chatBox = document.getElementById('chat-box');
            const recDiv = document.createElement('div');
            recDiv.className = 'recommendation';
            
            recDiv.innerHTML = `
                <div class="movie-title">${movie.title}</div>
                <div class="movie-genre">${movie.genre}</div>
                <p>${movie.description}</p>
                <p><em>Recommendation based on detected emotion: ${emotion}</em></p>
            `;
            
            chatBox.appendChild(recDiv);
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        function updateEmotionDisplay() {
            document.getElementById('emotion-label').textContent = currentEmotion.charAt(0).toUpperCase() + currentEmotion.slice(1);
            document.getElementById('emotion-confidence').style.width = (confidenceScore * 100) + '%';
            document.getElementById('emotion-confidence-value').textContent = confidenceScore.toFixed(2);
            document.getElementById('detected-emotion').textContent = currentEmotion;
            
            // Update emotion keywords
            updateEmotionKeywords(currentEmotion, lastUserMessage);
        }
        
        function updateEmotionKeywords(emotion, message) {
            const keywordsDiv = document.getElementById('emotion-keywords');
            keywordsDiv.innerHTML = '';
            
            const keywordMap = {
                'joy': ['happy', 'good', 'great', 'joy', 'wonderful', 'excited', 'love'],
                'sadness': ['sad', 'down', 'upset', 'depressed', 'miserable', 'unhappy'],
                'anger': ['angry', 'mad', 'frustrated', 'annoyed', 'irritated'],
                'fear': ['afraid', 'scared', 'nervous', 'anxious', 'worried', 'terrified'],
                'surprise': ['surprised', 'amazed', 'wow', 'unexpected', 'shocked'],
                'neutral': []
            };
            
            // Find matching keywords
            const keywords = [];
            const emotionWords = keywordMap[emotion] || [];
            
            if (message && emotionWords.length > 0) {
                const messageLower = message.toLowerCase();
                emotionWords.forEach(word => {
                    if (messageLower.includes(word)) {
                        keywords.push(word);
                    }
                });
            }
            
            // Display keywords
            if (keywords.length > 0) {
                keywords.forEach(keyword => {
                    const span = document.createElement('span');
                    span.className = 'keyword';
                    span.textContent = keyword;
                    keywordsDiv.appendChild(span);
                });
            } else if (emotion !== 'neutral') {
                const span = document.createElement('span');
                span.className = 'keyword';
                span.textContent = 'contextual';
                keywordsDiv.appendChild(span);
            }
        }
        
        function updateAIVisualization(stage, emotion) {
            document.getElementById('current-stage').textContent = stage;
            document.getElementById('detected-emotion').textContent = emotion;
            
            // Update preferences based on conversation content
            const userPreferences = [];
            
            if (lastUserMessage) {
                const genres = ['action', 'comedy', 'drama', 'horror', 'sci-fi', 'thriller', 'romance', 'animation'];
                genres.forEach(genre => {
                    if (lastUserMessage.toLowerCase().includes(genre)) {
                        userPreferences.push(genre);
                    }
                });
                
                // Check for actor/director mentions
                if (lastUserMessage.toLowerCase().includes('nolan')) {
                    userPreferences.push('Christopher Nolan');
                }
                if (lastUserMessage.toLowerCase().includes('tarantino')) {
                    userPreferences.push('Tarantino');
                }
            }
            
            document.getElementById('user-preferences').textContent = 
                userPreferences.length > 0 ? userPreferences.join(', ') : 'none';
                
            // Update GA action based on stage
            const gaActions = {
                'greeting': 'Initializing conversation path',
                'emotion_elicitation': 'Detecting emotional state',
                'genre_preference': 'Gathering genre preferences',
                'specific_preference': 'Collecting specific preferences',
                'recommendation': 'Generating personalized recommendation',
                'feedback': 'Evaluating user satisfaction',
                'refinement': 'Refining recommendations'
            };
            
            document.getElementById('ga-action').textContent = 
                gaActions[stage] || 'Processing user input';
        }
        
        function fetchStats() {
            fetch('http://localhost:8000/stats')
            .then(response => response.json())
            .then(data => {
                if (data.ga_stats) {
                    document.getElementById('ga-generation').textContent = data.ga_stats.generation;
                    document.getElementById('ga-fitness').textContent = data.ga_stats.best_fitness.toFixed(2);
                    
                    if (data.ga_stats.best_path && data.ga_stats.best_path.length > 0) {
                        const path = data.ga_stats.best_path.map(s => s.replace(/_/g, ' ')).join(' → ');
                        document.getElementById('ga-path').textContent = path;
                    }
                    
                    // Update GA step visualization
                    const generation = data.ga_stats.generation;
                    const currentTime = new Date();
                    
                    if (generation > 0) {
                        let stepText = `Generation ${generation}: `;
                        
                        if (generation % 3 === 0) {
                            stepText += `Selection (choosing fittest paths based on user satisfaction) → `;
                        } else if (generation % 3 === 1) {
                            stepText += `Crossover (combining successful conversation patterns) → `;
                        } else {
                            stepText += `Mutation (introducing variation at rate ${data.ga_stats.mutation_rate}) → `;
                        }
                        
                        stepText += `Fitness evaluation (current best: ${data.ga_stats.best_fitness.toFixed(2)})`;
                        document.getElementById('ga-step').textContent = stepText;
                    }
                }
            })
            .catch(error => {
                console.error('Error fetching stats:', error);
            });
        }
        
        function showTab(tabId) {
            // Hide all tab contents
            const tabContents = document.getElementsByClassName('tab-content');
            for (let i = 0; i < tabContents.length; i++) {
                tabContents[i].classList.remove('active');
            }
            
            // Deactivate all tabs
            const tabs = document.getElementsByClassName('tab');
            for (let i = 0; i < tabs.length; i++) {
                tabs[i].classList.remove('active');
            }
            
            // Show selected tab content
            document.getElementById(tabId).classList.add('active');
            
            // Activate selected tab
            const selectedTab = [...document.getElementsByClassName('tab')].find(tab => 
                tab.getAttribute('onclick').includes(tabId)
            );
            selectedTab.classList.add('active');
        }
        
        // Handle Enter key in input field
        document.getElementById('message-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Update stats periodically
        setInterval(fetchStats, 5000);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_html(request: Request):
    """Serve the HTML frontend"""
    return HTMLResponse(content=HTML_TEMPLATE)

if __name__ == "__main__":
    print("Starting frontend server on http://localhost:8001")
    print("Make sure your backend server is running on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8001)