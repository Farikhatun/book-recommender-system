from flask import Flask, request, jsonify, render_template
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load data
books = pd.read_csv('books.csv', usecols=['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-L'])
ratings = pd.read_csv('ratings.csv', usecols=['User-ID', 'ISBN', 'Book-Rating'])

# Membersihkan data
ratings_filtered = ratings[ratings['Book-Rating'] > 0]
combined_data = pd.merge(books, ratings_filtered, on='ISBN')
combined_data = combined_data.dropna(subset=['Book-Title'])

# Membuat rating count
book_rating_count = combined_data.groupby('Book-Title')['Book-Rating'].count().reset_index()
book_rating_count = book_rating_count.rename(columns={'Book-Rating': 'totalRatingCount'})

# Filter popular books
popularity_threshold = 250
popular_books = combined_data.merge(book_rating_count, on='Book-Title')
popular_books = popular_books[popular_books['totalRatingCount'] >= popularity_threshold]

# Pivot table untuk fitur buku
book_features_df = popular_books.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating', fill_value=0)
book_features_matrix = csr_matrix(book_features_df.values)

# Model KNN
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(book_features_matrix)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['GET'])
def recommend():
    book_title = request.args.get('title')
    if book_title not in book_features_df.index:
        random_book = book_features_df.sample(1).index[0]
        query_index = book_features_df.index.get_loc(random_book)
        distances, indices = model_knn.kneighbors(book_features_df.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)

        recommendations = []
        for i in range(1, len(distances.flatten())):
            recommended_title = book_features_df.index[indices.flatten()[i]]
            image_url = popular_books.loc[popular_books['Book-Title'] == recommended_title, 'Image-URL-L'].values[0]
            author = popular_books.loc[popular_books['Book-Title'] == recommended_title, 'Book-Author'].values[0]
            year = popular_books.loc[popular_books['Book-Title'] == recommended_title, 'Year-Of-Publication'].values[0]
            publisher = popular_books.loc[popular_books['Book-Title'] == recommended_title, 'Year-Of-Publication'].values[0]
            recommendations.append({
                'title': recommended_title,
                'image_url': image_url,
                'author': author,
                'year': year
            })
        return jsonify({
            'message': f"Buku tidak ditemukan di dalam data, coba rekomendasi kami berdasarkan buku '{random_book}'",
            'recommendations': recommendations
        })

    query_index = book_features_df.index.get_loc(book_title)
    distances, indices = model_knn.kneighbors(book_features_df.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)

    recommendations = []
    for i in range(1, len(distances.flatten())):
        recommended_title = book_features_df.index[indices.flatten()[i]]
        book_details = popular_books.loc[popular_books['Book-Title'] == recommended_title].iloc[0]

        recommendations.append({
            'title': recommended_title,
            'author': book_details['Book-Author'],
            'year': book_details['Year-Of-Publication'],
            'publisher': book_details['Publisher'],
            'image_url': book_details['Image-URL-L'],
            'distance': distances.flatten()[i]
        })

    return jsonify({'recommendations': recommendations})


if __name__ == '__main__':
    app.run(debug=True)
