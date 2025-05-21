import pandas as pd
import numpy as np
import re
import logging 
import os
from typing import List, Dict, Union, Optional
from datetime import datetime 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import MinMaxScaler
from fuzzywuzzy import fuzz

# Constants that control the recommendation system behavior
MIN_RATING_THRESHOLD = 3.0  
MAX_RECOMMENDATIONS = 10    
DIVERSITY_PENALTY = 0.3     
POPULARITY_WEIGHT = 0.2     
RECENCY_WEIGHT = 0.15       

class BookRecommender:
    def __init__(self, data_path:str = "data/Books_Data_Clean.csv"):
        # Initialize the recommender system with a path to the book data
        self.logger = self._setup_logger()
        self.df = self._load_and_preprocess_data(data_path)
        # Weights for different features when calculating similarity
        self.feature_weights = {
            'genre': 0.4,      # Genre has highest importance
            'author': 0.3,     # Author is second most important
            'language': 0.1,   # Language has lower importance
            'description': 0.2 # Description has medium importance
        }
        self.similarity_matrix = None
        self._initialize_models()
        
    
        self.history_dir = "recommendation_history"
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)
        
        # Initialize or load recommendation history
        self.history_file = os.path.join(self.history_dir, "recommendation_history.csv")
        self._initialize_history()

    def _setup_logger(self):
        # Set up logging to track system operations and errors
        logging.basicConfig(
            level = logging.INFO,
            format = '%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _load_and_preprocess_data(self, data_path: str) -> pd.DataFrame:
        # Load the book data from CSV and prepare it for analysis
        try:
            df = pd.read_csv(data_path)
            df = self._clean_data(df)
            df = self._engineer_features(df)  #creates new useful columns (features) based on the existing data
            df = self._normalize_features(df)  
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
        
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        # Clean and standardize the book data by handling missing values and formatting text
        # Fill missing values with sensible defaults
        df['description'] = df.get('description', pd.Series([' '] * len(df))).fillna('')
        df['genre'] = df.get('genre', 'Unknown').fillna('Unknown')
        df['author'] = df.get('Author', 'Unknown').fillna('Unknown')
        df['language'] = df.get('language_code', 'English').fillna('English')
        df['title'] = df.get('Book Name', 'Unknown Title').fillna('Unknown Title')

        # Keep original title for display purposes
        df['display_title'] = df['title']

        # Clean text fields by removing special characters and converting to lowercase
        # This helps in better text matching and comparison
        text_columns = ['title', 'author', 'genre', 'description']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: re.sub(r'[^\w\s]', '', str(x).lower()))

        # Convert numerical columns to proper numeric types
        if 'Publishing Year' in df.columns:
            df['publication_year'] = pd.to_numeric(df['Publishing Year'], errors = 'coerce')

            # Convert rating and sales data to numeric values
            df['Book_average_rating'] = pd.to_numeric(df.get('Book_average_rating', 0), errors = 'coerce')
            df['Book_ratings_count'] = pd.to_numeric(df.get('Book_ratings_count', 0), errors = 'coerce')
            df['gross sales'] = pd.to_numeric(df.get('gross sales', 0), errors ='coerce')

        return df
    
    def _engineer_features(self, df:pd.DataFrame) -> pd.DataFrame:
        # Create new features by combining existing data and calculating scores
        # Combine text features for similarity calculation
        df['combined_features'] = (
            df.get('genre', '') + ' ' +
            df.get('author', '') + ' ' +
            df.get('language', '') + ' ' +
            df.get('description', '').str[:200]  # Use first 200 chars of description to keep it manageable
        )

        # Calculate average rating, defaulting to 0 if missing
        df['average_rating'] = df['Book_average_rating'].fillna(0)

        # Calculate popularity score based on number of ratings and average rating
        # Using log1p to handle books with very high number of ratings
        if 'Book_ratings_count' in df.columns and 'Book_average_rating' in df.columns:
            df['popularity_score'] = (
                np.log1p(df['Book_ratings_count']) * df['Book_average_rating']
            )

        # Calculate recency score based on publication year
        # More recent books get higher scores
        if 'publication_year' in df.columns:
            current_year = datetime.now().year
            df['recency_score'] = 1 / (1 + (current_year - df['publication_year'].fillna(current_year)))

        return df
    
    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Scale numerical features to a range between 0 and 1
        # This ensures all features contribute equally to similarity calculations
        scaler = MinMaxScaler()
        if 'popularity_score' in df.columns:
            df['popularity_score'] = scaler.fit_transform(df[['popularity_score']])
        if 'recency_score' in df.columns:
            df['recency_score'] = scaler.fit_transform(df[['recency_score']])
        return df

    def _initialize_models(self):
        # Set up the recommendation models and calculate similarity between books
        # Create TF-IDF vectorizer for text similarity
        self.tfidf = TfidfVectorizer(
            stop_words='english',  # Remove common English words
            ngram_range=(1, 2),    # Consider both single words and pairs
            max_features=5000      # Limit vocabulary size
        )
        # Convert text features to TF-IDF matrix
        tfidf_matrix = self.tfidf.fit_transform(self.df['combined_features'])
        # Calculate similarity between all books
        self.similarity_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)

        # Enhance similarity scores with additional factors
        if 'popularity_score' in self.df.columns:
            self._enhance_with_popularity()
        if 'recency_score' in self.df.columns:
            self._enhance_with_recency()

    def _enhance_with_popularity(self):
        # Adjust similarity scores based on book popularity
        # Popular books get a boost in their similarity scores
        scores = self.df['popularity_score'].values
        for i in range(len(self.similarity_matrix)):
            self.similarity_matrix[i] = (
                (1 - POPULARITY_WEIGHT) * self.similarity_matrix[i] +  # Original similarity
                POPULARITY_WEIGHT * scores                            # Popularity boost
            )

    def _enhance_with_recency(self):
        # Adjust similarity scores based on book publication year
        # Recent books get a boost in their similarity scores
        scores = self.df['recency_score'].values
        for i in range(len(self.similarity_matrix)):
            self.similarity_matrix[i] = (
                (1 - RECENCY_WEIGHT) * self.similarity_matrix[i] +  # Original similarity
                RECENCY_WEIGHT * scores                            # Recency boost
            )

    def _find_book_index(self, book_title: str) -> Optional[int]:
        # Find the index of a book in the dataset using exact or fuzzy matching
        # First try exact match
        exact_match = self.df[self.df['title'].str.lower() == book_title.lower()]
        if not exact_match.empty:
            return exact_match.index[0]
        
        # If no exact match, try fuzzy matching with similarity threshold of 70%
        fuzzy_scores = [
            (i, fuzz.ratio(book_title.lower(), row['title']))
            for i, row in self.df.iterrows()
        ]
        fuzzy_scores.sort(key=lambda x: x[1], reverse=True)

        return fuzzy_scores[0][0] if fuzzy_scores[0][1] >= 70 else None

    def _apply_diversity_penalty(self, scores: List[tuple]) -> List[tuple]:
        # Adjust recommendation scores to ensure diverse recommendations
        # This prevents recommending very similar books
        if len(scores) <= 1:
            return scores
        diversified = [scores[0]]
        for i in range(1, len(scores)):
            idx = scores[i][0]
            # Calculate minimum similarity with already selected books
            min_sim = min(self.similarity_matrix[idx][prev_idx] for prev_idx, _ in diversified)
            # Apply penalty based on similarity with previous recommendations
            penalized_score = scores[i][1] * (1 - DIVERSITY_PENALTY * min_sim)
            diversified.append((idx, penalized_score))
        return diversified

    def _initialize_history(self):
        """Initialize or load the recommendation history file"""
        if not os.path.exists(self.history_file):
            # Create new history file with headers
            history_df = pd.DataFrame(columns=[
                'timestamp',
                'source_book',
                'recommended_book',
                'author',
                'genre',
                'rating',
                'similarity_score'
            ])
            history_df.to_csv(self.history_file, index=False)
            self.logger.info(f"Created new recommendation history file at {self.history_file}")
        else:
            self.logger.info(f"Loaded existing recommendation history from {self.history_file}")

    def save_to_history(self, source_book: str, recommendations: List[Dict]):
        """Save recommendations to history file"""
        try:
            # Create new history entries
            new_entries = []
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            for rec in recommendations:
                entry = {
                    'timestamp': timestamp,
                    'source_book': source_book,
                    'recommended_book': rec['title'],
                    'author': rec['author'],
                    'genre': rec.get('genre', 'N/A'),
                    'rating': rec.get('rating', 'N/A'),
                    'similarity_score': rec.get('similarity_score', 0)
                }
                new_entries.append(entry)
            
            # Append to history file
            history_df = pd.DataFrame(new_entries)
            history_df.to_csv(self.history_file, mode='a', header=False, index=False)
            self.logger.info(f"Saved {len(new_entries)} recommendations to history")
            
        except Exception as e:
            self.logger.error(f"Error saving to history: {str(e)}")

    def get_recommendation_history(self, 
                                 source_book: Optional[str] = None,
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None) -> pd.DataFrame:
        """Get recommendation history with optional filters"""
        try:
            # Read history file
            history_df = pd.read_csv(self.history_file)
            
            # Apply filters if provided
            if source_book:
                history_df = history_df[history_df['source_book'] == source_book]
            
            if start_date:
                history_df = history_df[history_df['timestamp'] >= start_date]
            
            if end_date:
                history_df = history_df[history_df['timestamp'] <= end_date]
            
            return history_df
            
        except Exception as e:
            self.logger.error(f"Error reading history: {str(e)}")
            return pd.DataFrame()

    def get_recommendations(
        self,
        book_title: str,
        num_recommendations: int = 5,
        min_rating: float = MIN_RATING_THRESHOLD,
        save_to_history: bool = True
    ) -> List[Dict[str, Union[str, float]]]:
        # Get book recommendations based on a given book title
        try:
            # Find the index of the input book
            idx = self._find_book_index(book_title)
            if idx is None:
                self.logger.warning(f"No match found for: {book_title}")
                return []

            # Get similarity scores for all books
            sim_scores = list(enumerate(self.similarity_matrix[idx]))

            # Filter out books with low ratings
            if 'average_rating' in self.df.columns:
                sim_scores = [
                    (i, score) for i, score in sim_scores
                    if self.df.iloc[i]['average_rating'] >= min_rating
                ]

            # Sort by similarity and apply diversity penalty
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = self._apply_diversity_penalty(sim_scores)
            top_indices = [i[0] for i in sim_scores[1:num_recommendations + 1]]

            # Get recommendations
            recommendations = [
                {
                    'title': self.df.iloc[i]['display_title'],
                    'author': self.df.iloc[i]['author'],
                    'rating': self.df.iloc[i]['average_rating'],
                    'genre': self.df.iloc[i].get('genre', 'N/A'),
                    'sales': self.df.iloc[i].get('gross sales', 'N/A'),
                    'similarity_score': round(sim_scores[i][1], 3)
                }
                for i in top_indices
            ]

            # Save to history if requested
            if save_to_history:
                self.save_to_history(book_title, recommendations)

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return []

    def get_similar_books_by_features(
        self,
        genre: Optional[str] = None,
        author: Optional[str] = None,
        language: Optional[str] = None,
        num_recommendations: int = 5
    ) -> List[Dict[str, Union[str, float]]]:
        # Find books similar to given features (genre, author, or language)
        try:
            # Combine provided features into a search query
            query_parts = filter(None, [genre, author, language])
            query_str = ' '.join([part.lower() for part in query_parts])

            if not query_str:
                return []

            # Convert query to TF-IDF vector and find similar books
            query_vector = self.tfidf.transform([query_str])
            cosine_sim = linear_kernel(query_vector, self.tfidf.transform(self.df['combined_features'])).flatten()
            top_indices = cosine_sim.argsort()[::-1][:num_recommendations]
            top_scores = [(i, cosine_sim[i]) for i in top_indices]

            # Return information about similar books
            return [
                {
                    'title': self.df.iloc[i]['title'],
                    'author': self.df.iloc[i]['author'],
                    'genre': self.df.iloc[i].get('genre', 'N/A'),
                    'similarity_score': round(score, 3)
                }
                for i, score in top_scores
            ]

        except Exception as e:
            self.logger.error(f"Error in feature-based recommendation: {str(e)}")
            return []

    def refresh_data(self, new_data_path: str):
        # Update the recommender system with new book data
        try:
            self.df = self._load_and_preprocess_data(new_data_path)
            self._initialize_models()
            self.logger.info("Data refreshed successfully.")
        except Exception as e:
            self.logger.error(f"Error refreshing data: {str(e)}")
            raise






















































































