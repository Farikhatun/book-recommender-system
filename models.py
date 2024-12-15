import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

def get_recommendations(title, books):
    # Membuat model rekomendasi
    book_features = books.pivot_table(
        values='Year-Of-Publication', index='Book-Title', columns='ISBN', fill_value=0
    )
    book_matrix = csr_matrix(book_features.values)
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(book_matrix)

    # Mendapatkan rekomendasi
    if title not in book_features.index:
        return []
    distances, indices = model_knn.kneighbors(
        book_features.loc[title].values.reshape(1, -1), n_neighbors=6
    )
    recommended_titles = [
        book_features.index[i] for i in indices.flatten() if book_features.index[i] != title
    ]
    return books[books['Book-Title'].isin(recommended_titles)].to_dict(orient='records')
