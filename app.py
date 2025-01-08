# app.py
from flask import Flask, render_template, request
import pandas as pd
from recommender import InteractiveMusicRecommender

app = Flask(__name__)

# Initialize the recommender system
df = pd.read_csv('dataset.csv')
recommender = InteractiveMusicRecommender(df=df)

@app.route('/')
def home():
    # Get list of unique genres for the dropdown
    genres = sorted(df['track_genre'].unique())
    moods = ['happy', 'sad', 'energetic', 'relaxed', 'party', 'focus']
    return render_template('index.html', genres=genres, moods=moods)

@app.route('/recommend', methods=['POST'])
def recommend():
    search_type = request.form.get('search_type')
    query = request.form.get('query')
    error_message = None
    recommendations = []
    
    if not query:
        error_message = "Please enter a search term"
    else:
        try:
            if search_type == 'song':
                recommendations = recommender.get_recommendations(query)
            elif search_type == 'artist':
                recommendations = recommender.get_recommendations_by_artist(query)
            elif search_type == 'genre':
                recommendations = recommender.get_recommendations_by_genre(query)
            elif search_type == 'mood':
                recommendations = recommender.get_recommendations_by_mood(query)
            
            if isinstance(recommendations, str):  # Error message
                error_message = recommendations
                recommendations = []
                
        except Exception as e:
            error_message = str(e)
            recommendations = []
    
    genres = sorted(df['track_genre'].unique())
    moods = ['happy', 'sad', 'energetic', 'relaxed', 'party', 'focus']
    return render_template('index.html', 
                         genres=genres,
                         moods=moods,
                         recommendations=recommendations,
                         error_message=error_message,
                         selected_type=search_type,
                         query=query)

if __name__ == '__main__':
    app.run(debug=True)