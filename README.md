# AI Movie Recommender with Genetic Algorithm

An emotion-driven movie recommendation system that uses genetic algorithms to optimize conversation flow and emotion analysis to personalize movie recommendations.

## 🚀 Features

- **Emotion Detection**: Real-time analysis of user emotions from text
- **Genetic Algorithm Optimization**: Conversation paths evolve based on user satisfaction
- **Personalized Recommendations**: Movies matched to emotional state and preferences
- **Performance Metrics**: Live statistics showing system learning and improvement
- **Interactive Chat Interface**: Real-time conversation with the AI recommender

## 🛠️ Tech Stack

- **Backend**: FastAPI with Python
- **Frontend**: HTML, CSS, JavaScript
- **AI Components**: Genetic Algorithms, NLP for emotion detection
- **Data**: MovieLens dataset (with fallback for deployment)

## 📦 Deployment Options

### Option 1: Railway (Recommended for Full-Stack)

1. **Fork this repository**
2. **Connect to Railway**:
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub
   - Click "Deploy from GitHub repo"
   - Select this repository

3. **Environment Variables** (if needed):
   ```
   PORT=8000
   PYTHON_VERSION=3.11
   ```

4. **Deploy**: Railway will automatically deploy using the `Procfile`

### Option 2: Render

1. **Fork this repository**
2. **Connect to Render**:
   - Go to [render.com](https://render.com)
   - Sign up with GitHub
   - Click "New Web Service"
   - Connect this repository

3. **Configuration**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3

### Option 3: Heroku

1. **Install Heroku CLI**
2. **Login and Create App**:
   ```bash
   heroku login
   heroku create your-app-name
   ```

3. **Deploy**:
   ```bash
   git push heroku main
   ```

### Option 4: Netlify (Frontend Only)

For static deployment of the frontend:

1. **Build the frontend**:
   ```bash
   # Copy templates/index.html to root
   cp templates/index.html index.html
   ```

2. **Deploy to Netlify**:
   - Drag and drop the project folder to netlify.com
   - Or connect via GitHub

## 🏃‍♂️ Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MuhammadKhan148/AI-Movie-Recommender-with-Genetic-Algorithm-.git
   cd AI-Movie-Recommender-with-Genetic-Algorithm-
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   # Option 1: Main FastAPI app
   uvicorn main:app --reload
   
   # Option 2: Alternative app
   uvicorn app:app --reload
   
   # Option 3: Frontend server
   python frontend.py
   ```

4. **Access the application**:
   - Open `http://localhost:8000` in your browser

## 📊 Project Structure

```
├── main.py                     # Main FastAPI application
├── app.py                      # Alternative FastAPI app
├── frontend.py                 # Frontend server
├── enhanced_genetic_recommender.py  # Core AI logic
├── genetic_recommender.py      # Original implementation
├── ml_emotion_detector.py      # ML-based emotion detection
├── templates/
│   └── index.html             # Main UI template
├── requirements.txt           # Python dependencies
├── Procfile                   # Deployment configuration
└── README.md                  # This file
```

## 🧠 How It Works

1. **User Interaction**: Users chat with the AI through a web interface
2. **Emotion Analysis**: System detects emotions from user text
3. **Genetic Algorithm**: Optimizes conversation paths based on user satisfaction
4. **Movie Recommendations**: Suggests movies based on emotional state and preferences
5. **Continuous Learning**: System improves through evolutionary optimization

## 🎯 Key Components

### Genetic Algorithm
- **Population**: Multiple conversation paths
- **Fitness Function**: User satisfaction scores
- **Selection**: Top-performing paths chosen for reproduction
- **Crossover**: Combining successful conversation strategies
- **Mutation**: Random variations for exploration

### Emotion Detection
- **Keyword-based**: Fast emotion detection from text patterns
- **ML-based**: Advanced transformer model for complex emotions
- **Genre Mapping**: Emotions mapped to appropriate movie genres

## 🔧 Configuration

### Environment Variables
- `PORT`: Server port (default: 8000)
- `USE_ML_EMOTIONS`: Enable/disable ML emotion detection (default: False)

### Data Files
- Large MovieLens dataset files are excluded from deployment
- System uses fallback movie database for production
- To use full dataset, download from [MovieLens](https://grouplens.org/datasets/movielens/)

## 📈 Performance

- **User Satisfaction**: Typically improves from 60% to 93+ over 20 generations
- **Response Time**: Sub-second response for most interactions
- **Scalability**: Designed for multiple concurrent users

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is created for educational purposes as part of an Applied AI course.

## 🎬 Demo

The system provides real-time movie recommendations based on:
- Current emotional state
- Genre preferences
- Previous interactions
- Optimized conversation flow

Try it live: [Your Deployment URL Here] 