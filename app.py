import streamlit as st
import pandas as pd
from scipy.sparse import load_npz
import pickle

def get_recommendations_by_title(original_title, knn_model, final_features, content_df, top_n=10):
    """
    Get the top N movie recommendations based on the original title using a pre-fitted KNN model.

    Args:
        original_title (str): The title of the movie for which to find recommendations.
        knn_model: The pre-fitted KNN model.
        final_features: The final feature matrix used in the KNN model.
        content_df: DataFrame containing 'original_title' and other movie data.
        top_n (int): Number of top recommendations to return.

    Returns:
        DataFrame: A DataFrame containing the top N similar movies with specified columns.
    """
    # Find the index of the movie that matches the original_title
    idx = content_df.index[content_df['original_title'] == original_title].tolist()

    if not idx:
        return f"Movie with title '{original_title}' not found in the dataset."

    # Get the indices of the top N most similar movies using KNN
    distances, indices = knn_model.kneighbors(final_features[idx], n_neighbors=top_n + 1)

    # Get the indices of the recommended movies (excluding the first one, which is the movie itself)
    recommended_indices = indices.flatten()[1:top_n + 1]
    recommended_distances = distances.flatten()[1:top_n + 1]

    # Create a DataFrame for the recommended movies
    recommended_df = content_df.iloc[recommended_indices].copy()

    # Add the similarity score (inverse of distance) to the DataFrame
    recommended_df['similarity_score'] = 1 - recommended_distances

    # Return the DataFrame with the desired columns
    return recommended_df[['id', 'original_title', 'vote_count', 'vote_average', 'similarity_score']]


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
        top_recommendations = get_recommendations_by_title(selected_movie, knn, final_features =feature_set_matrix , content_df = content_df, top_n=10)
        top_5_recommendations = top_recommendations[['id', 'original_title', 'vote_count', 'vote_average', 'similarity_score']].head(5)

        # Display the top 5 recommendations
        st.write(f"Top 5 movie recommendations for: {selected_movie}")
        st.dataframe(top_5_recommendations)

if __name__ == "__main__":
    main()
