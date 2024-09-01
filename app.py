import streamlit as st
import pandas as pd
from scipy.sparse import load_npz
import pickle
from PIL import Image
import os

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

def DisplayMoviePosterStreamlit(top_recommendations, posters_folder):
    """
    Display the movie posters of the top recommended movies using Streamlit.

    Parameters:
    - top_recommendations (DataFrame): A DataFrame containing the top movie recommendations, which includes the 'id' column.
    - posters_folder (str): The path to the folder containing the movie posters named according to 'id' in the format "id.jpg".

    Returns:
    - None: Displays the posters in a 2x5 grid using Streamlit.
    """

    # Determine sorting behavior based on available columns
    if {'similarity_score', 'pred'}.issubset(top_recommendations.columns):
        top_recommendations = top_recommendations.sort_values(by=['similarity_score', 'pred'], ascending=[False, False])
    elif 'similarity_score' in top_recommendations.columns:
        top_recommendations = top_recommendations.sort_values(by='similarity_score', ascending=False)

    # Create columns for a 2x5 grid
    cols = st.columns(5)

    for i, (index, row) in enumerate(top_recommendations.iterrows()):
        imdb_id = row['id']
        poster_path = os.path.join(posters_folder, f"{imdb_id}.jpg")

        # Load the image
        try:
            img = Image.open(poster_path).convert('RGB')
        except FileNotFoundError:
            st.write(f"Poster not found for IMDb ID {imdb_id}.")
            img = Image.new('RGB', (500, 750), color=(73, 109, 137))  # Placeholder image if not found

        # Display the image in the grid
        with cols[i % 5]:
            st.image(img, caption=row['original_title'], use_column_width=True)

        # Move to next row if we've filled the current row
        if (i + 1) % 5 == 0:
            cols = st.columns(5)

def main():
    st.title("Movie Recommendation System")

    # Introduction text
    st.write("Recommender systems are essential tools for enhancing user experience, driving engagement, increasing sales, and providing personalized content in various domains, from e-commerce and streaming services to news platforms and social networks.")
    st.write("In this app, you can search for a movie you have watched and liked, and I will recommend five more movies based on content-based filtering model. Hopefully, you have not watched all five!")
    
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
        top_recommendations = get_recommendations_by_title(selected_movie, knn, final_features=feature_set_matrix, content_df=content_df, top_n=10)
        top_5_recommendations = top_recommendations[['id', 'original_title', 'vote_count', 'vote_average', 'similarity_score']].head(5)
        data_table = top_5_recommendations[['original_title', 'vote_count', 'vote_average', 'similarity_score']]

        # Add short descriptions
        st.write("**Vote Count:** The total number of votes the movie has received.")
        st.write("**Vote Average:** The average rating of the movie based on the votes it has received.")
        st.write("**Similarity Score:** A measure of how similar the recommended movie is to the selected movie (closer to 1 indicates higher similarity).")

        # Display the top 5 recommendations
        st.write(f"Top 5 movie recommendations for: {selected_movie}")
        st.dataframe(data_table)

        # Display posters
        posters_folder = os.path.join(os.getcwd(), 'posters')
        DisplayMoviePosterStreamlit(top_5_recommendations, posters_folder)

    # Footer
    st.write("---")
    st.write("**Created By:** Mthokozisi Nsango")
    st.write("**Last data update:** July 2024")

if __name__ == "__main__":
    main()
