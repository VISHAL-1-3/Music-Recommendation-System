import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

class InteractiveMusicRecommender:
    def __init__(self, data_path=None, df=None):
        if df is not None:
            self.data = df
        else:
            self.data = pd.read_csv(data_path)
            
        # Define feature weights
        self.feature_weights = {
            'popularity': 0.5,
            'danceability': 1.0,
            'energy': 1.0,
            'key': 0.3,
            'loudness': 0.7,
            'mode': 0.3,
            'speechiness': 0.8,
            'acousticness': 1.0,
            'instrumentalness': 1.0,
            'liveness': 0.6,
            'valence': 1.0,
            'tempo': 0.7,
            'time_signature': 0.3
        }
        
        self.features = list(self.feature_weights.keys())
        self.prepare_data()
        
    def prepare_data(self):
        """Prepare data with weighted features"""
        self.songs = self.data.copy()
        
        # Standardize features
        self.scaler = StandardScaler()
        self.songs[self.features] = self.scaler.fit_transform(self.songs[self.features])
        
        # Apply feature weights
        for feature, weight in self.feature_weights.items():
            self.songs[feature] *= weight
            
        # Pre-compute genre weights
        self.genre_weights = pd.get_dummies(self.songs['track_genre'])
        
    def get_recommendations(self, song_name, n_recommendations=5):
        """Get recommendations based on song name with optimized similarity calculation"""
        song_name_lower = song_name.lower()
        
        # Find exact matches (case-insensitive)
        exact_matches = self.songs[self.songs['track_name'].str.lower() == song_name_lower]
        
        if len(exact_matches) == 0:
            print(f"\nNo exact matches found for '{song_name}'. Showing similar songs:")
            similar_songs = self.songs[self.songs['track_name'].str.lower().str.contains(song_name_lower, na=False)]
            if len(similar_songs) == 0:
                return f"No songs found matching '{song_name}'"
            reference_song = similar_songs.iloc[0]
        else:
            print(f"\nFound exact match:")
            reference_song = exact_matches.iloc[0]
        
        # Calculate similarities using vectorized operations
        feature_similarities = cosine_similarity(
            self.songs[self.features],
            reference_song[self.features].values.reshape(1, -1)
        ).flatten()
        
        # Calculate genre similarity using one-hot encoded genres
        genre_similarity = (self.genre_weights[self.songs['track_genre'] == reference_song['track_genre']]).values.flatten()
        
        # Combine similarities (70% features, 30% genre)
        similarities = (0.7 * feature_similarities) + (0.3 * genre_similarity)
        
        # Create recommendations list
        recommendations = []
        
        # First add exact matches if they exist
        if len(exact_matches) > 0:
            for _, match in exact_matches.iterrows():
                recommendations.append({
                    'track_name': match['track_name'],
                    'artists': match['artists'],
                    'track_genre': match['track_genre'],
                    'similarity_score': 1.0,
                    'match_type': 'Exact Match'
                })
        
        # Then add similar songs (excluding exact matches)
        added_songs = set((exact_matches['track_name'] + exact_matches['artists']).values)
        
        # Get top indices excluding exact matches
        mask = ~self.songs.index.isin(exact_matches.index)
        filtered_similarities = similarities.copy()
        filtered_similarities[~mask] = -1
        top_indices = np.argsort(filtered_similarities)[::-1][:n_recommendations]
        
        for idx in top_indices:
            song = self.songs.iloc[idx]
            recommendations.append({
                'track_name': song['track_name'],
                'artists': song['artists'],
                'track_genre': song['track_genre'],
            })
        
        return recommendations
    
    def get_recommendations_by_artist(self, artist_name, n_recommendations=10):
      """Get recommendations based on artist name, prioritizing all songs by the artist"""
      artist_name_lower = artist_name.lower()
    
    # Find all songs by the artist (case-insensitive)
      artist_songs = self.songs[self.songs['artists'].fillna('').str.lower().str.contains(artist_name_lower, na=False)]
    
      if len(artist_songs) == 0:
        return f"Artist '{artist_name}' not found in the dataset."
    
    # Remove duplicates based on track name
      artist_songs = artist_songs.drop_duplicates(subset=['track_name'], keep='first')
      print(f"\nFound {len(artist_songs)} unique songs by {artist_name}")
    
    # Create recommendations list
      recommendations = []
    
    # Add all songs by the artist
      for _, song in artist_songs.iterrows():
        recommendations.append({
            'track_name': song['track_name'],
            'artists': song['artists'],
            'track_genre': song['track_genre'],
            'similarity_score': 1.0,
            'match_type': 'Artist Song'
        })
    
    # If we don't have enough songs by the artist to meet n_recommendations,
    # only then add similar songs by other artists
      if len(recommendations) < n_recommendations:
        # Calculate average features of artist's songs
        artist_features = artist_songs[self.features].mean().values.reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(self.songs[self.features], artist_features).flatten()
        
        # Exclude songs by the searched artist
        mask = ~self.songs['artists'].fillna('').str.lower().str.contains(artist_name_lower, na=False)
        filtered_similarities = similarities.copy()
        filtered_similarities[~mask] = -1
        top_indices = np.argsort(filtered_similarities)[::-1]
        
        seen_songs = set((song['track_name'], song['artists']) for song in recommendations)
        
        for idx in top_indices:
            if len(recommendations) >= n_recommendations:
                break
                
            song = self.songs.iloc[idx]
            song_key = (song['track_name'], song['artists'])
            
            if song_key not in seen_songs:
                recommendations.append({
                    'track_name': song['track_name'],
                    'artists': song['artists'],
                    'track_genre': song['track_genre'],
                    'match_type': 'Similar Song'
                })
                seen_songs.add(song_key)
    
      return recommendations[:n_recommendations]
    
    def get_recommendations_by_genre(self, genre, n_recommendations=5):
        """Get recommendations based on genre"""
        # Find songs in the genre
        genre_songs = self.songs[self.songs['track_genre'].str.lower() == genre.lower()]
        if len(genre_songs) == 0:
            return f"Genre '{genre}' not found in the dataset."
        
        # Get average features of the genre
        genre_features = genre_songs[self.features].mean().values.reshape(1, -1)
        
        # Calculate similarities
        similarities = cosine_similarity(self.songs[self.features], genre_features).flatten()
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        
        return [
            {
                'track_name': self.songs.iloc[idx]['track_name'],
                'artists': self.songs.iloc[idx]['artists'],
                'track_genre': self.songs.iloc[idx]['track_genre'],
                
            }
            for idx in top_indices
        ]
        
    def get_recommendations_by_mood(self, mood, n_recommendations=5):
        """Get recommendations based on mood"""
        mood_features = {
            'happy': {'valence': 0.8, 'energy': 0.7, 'danceability': 0.7},
            'sad': {'valence': 0.3, 'energy': 0.4, 'acousticness': 0.7},
            'energetic': {'energy': 0.8, 'tempo': 0.7, 'danceability': 0.7},
            'relaxed': {'energy': 0.3, 'acousticness': 0.7, 'instrumentalness': 0.5},
            'party': {'danceability': 0.8, 'energy': 0.8, 'valence': 0.7},
            'focus': {'instrumentalness': 0.6, 'energy': 0.5, 'valence': 0.5}
        }
        
        if mood.lower() not in mood_features:
            return f"Mood not recognized. Available moods: {', '.join(mood_features.keys())}"
            
        # Create target features
        target_features = np.zeros(len(self.features))
        for i, feature in enumerate(self.features):
            if feature in mood_features[mood.lower()]:
                target_features[i] = mood_features[mood.lower()][feature]
            else:
                target_features[i] = self.songs[feature].mean()
        
        # Calculate similarities
        similarities = cosine_similarity(self.songs[self.features], target_features.reshape(1, -1)).flatten()
        
        # Get top recommendations
        top_indices = np.argsort(similarities)[::-1][:n_recommendations]
        
        return [
            {
                'track_name': self.songs.iloc[idx]['track_name'],
                'artists': self.songs.iloc[idx]['artists'],
                'track_genre': self.songs.iloc[idx]['track_genre'],
                
            }
            for idx in top_indices
        ]