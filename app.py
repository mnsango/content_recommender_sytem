import streamlit as st
import pandas as pd
from scipy.sparse import load_npz
import pickle

def GetMovieRecommendation(original_title, content_df, feature_set_matrix, knn, n_neighbors=10):
    """
    Get movie recommendations based on a given movie title.

    Parameters:
    - original_title (str): The title of the movie for which recommendations are sought.
    - content_df (DataFrame): The DataFrame containing movie details.
    - feature_set_matrix (array-like): The feature matrix used for the KNN model.
    - knn (NearestNeighbors): The trained KNN model.
    - n_neighbors (int): The number of similar movies to return (default is 10).

    Returns:
    - DataFrame: A DataFrame containing the top N similar movies.
    """

    # Find the index of the movie in the DataFrame
    try:
        movie_idx = content_df.index[content_df['original_title'] == original_title][0]
    except IndexError:
        return f"Movie '{original_title}' not found in the dataset."

    # Find the nearest neighbors (similar movies)
    distances, indices = knn.kneighbors([feature_set_matrix[movie_idx]], n_neighbors=n_neighbors + 1)

    # Extract the indices and distances of the neighbors
    neighbor_indices = indices[0]
    neighbor_distances = distances[0]

    # Create a DataFrame for the neighbors
    neighbor_df = content_df.iloc[neighbor_indices].copy()

    # Calculate and add the similarity score (inverse of distance) to the DataFrame
    neighbor_df['similarity_score'] = 1 - neighbor_distances

    # Exclude the original movie
    neighbor_df = neighbor_df[neighbor_df.index != movie_idx]

    # Sort the DataFrame by similarity score
    neighbor_df = neighbor_df.sort_values(by='similarity_score', ascending=False)

    # Select the top N neighbors
    top_n_df = neighbor_df.head(n_neighbors)

    # Return the top N similar movies
    return top_n_df[['id', 'original_title', 'vote_count', 'vote_average', 'similarity_score']]

def main():
    st.title("Movie Recommendation System")

    # Load your data and model (assume these are pre-loaded or loaded from files)
    content_df = pd.read_parquet('content_based_features.parquet')
    feature_set_matrix = load_npz('content_based_sparse_feature_set.npz')

    with open('content_based_knn_model.pkl', 'rb') as file:
        knn = pickle.load(file)

    # Assisted search with text input
    search_term = st.text_input("Search for a movie to get recommendations")

    # Filter the movie list based on the search term
    if search_term:
        movie_list = content_df['original_title'].unique().tolist()
        filtered_movies = [movie for movie in movie_list if search_term.lower() in movie.lower()]

        # Let the user select a movie from the filtered list
        if filtered_movies:
            selected_movie = st.selectbox("Select a movie from the filtered results", filtered_movies)
        else:
            st.write("No movies found. Please refine your search.")
            selected_movie = None
    else:
        selected_movie = None

    # When the user selects a movie
    if selected_movie:
        top_recommendations = GetMovieRecommendation(selected_movie, content_df, feature_set_matrix, knn, n_neighbors=10)
        top_5_recommendations = top_recommendations[['id', 'original_title', 'vote_count', 'vote_average', 'similarity_score']].head(5)

        # Display the top 5 recommendations
        st.write(f"Top 5 movie recommendations for: {selected_movie}")
        st.dataframe(top_5_recommendations)

if __name__ == "__main__":
    main()
