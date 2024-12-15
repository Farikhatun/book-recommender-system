# Databricks notebook source
# DBTITLE 1,Mengimpor Library yang dibutuhkan
from pyspark.sql import SparkSession
from pyspark import SparkFiles
import pyspark.pandas as ps
import pandas as pd

ps.set_option('compute.ops_on_diff_frames', True)

# COMMAND ----------

# DBTITLE 1,Membuat Sesi Baru
spark = SparkSession.builder.master("local[*]")\
                            .appName("Recommender System")\
                            .getOrCreate()

spark.conf.set("spark.sql.repl.eagerEval.enabled", True)

spark

# COMMAND ----------

# DBTITLE 1,Menyiapkan Data
spark.sparkContext.addFile("https://yudiantosujana.com/files/books/books.csv")
spark.sparkContext.addFile("https://yudiantosujana.com/files/books/ratings.csv")

books = pd.read_csv("file://"+SparkFiles.get('books.csv'), usecols=['ISBN', 'Book-Title' , 'Book-Author', 'Year-Of-Publication', 'Publisher'],dtype={'ISBN':'str', 'Book-Title':'str' , 'Book-Author':'str', 'Year-Of-Publication':'str', 'Publisher':'str'})

ratings = pd.read_csv("file://"+SparkFiles.get('ratings.csv'), usecols=['User-ID', 'ISBN', 'Book-Rating'], dtype={'User-ID': 'int32', 'ISBN': 'str', 'Book-Rating': 'float32'})
print(len(ratings))
print(len(books))

# COMMAND ----------

# DBTITLE 1,Membersihkan Data
# Menghapus data yang memiliki rating 0
ratings_filtered = ratings.query('`Book-Rating` > 0')
ratings = ratings_filtered

# COMMAND ----------

# DBTITLE 1,Mengubah dataset ke pandas spark
# Convert ke dataset spark pandas
book_df = ps.DataFrame(books)
rating_df = ps.DataFrame(ratings)

# COMMAND ----------

book_df.head()

# COMMAND ----------

rating_df.head()

# COMMAND ----------

# DBTITLE 1,Menggabungkan dataset buku dan rating
df = ps.merge(book_df, rating_df,on='ISBN')
df.head()

# COMMAND ----------


# Hapus data yang titlenya kosong
combine_book_rating = df.dropna(axis = 0, subset = ['Book-Rating'])

# Menghitung jumlah rating untuk masing-masing buku
book_ratingCount = (combine_book_rating.
     groupby(by = ['Book-Title'])['Book-Rating'].
     count().
     reset_index().
     rename(columns = {'Book-Rating': 'totalRatingCount'})
     [['Book-Title', 'totalRatingCount']]
    )
book_ratingCount.head()

# COMMAND ----------

# Menggabungkan dataset combine_movie_rating dan movie_ratingCount
rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'Book-Title', right_on = 'Book-Title', how = 'left')
rating_with_totalRatingCount.head()

# COMMAND ----------

# Menampilkan statistik dataset
pd.options.display.float_format = '{:.2f}'.format
print(book_ratingCount['totalRatingCount'].describe())

# COMMAND ----------

# Menfilter film yang jumlah totalRatingCount minimal 250 agar tidak terlalu besar
popularity_threshold = 250
rating_popular_book= rating_with_totalRatingCount.query(f'totalRatingCount >= {popularity_threshold}')
rating_popular_book.head()

# COMMAND ----------

rating_popular_book.shape

# COMMAND ----------

# Membuat Pivot Table yang berisi kolom judul buku dan rating yang diberikan oleh setiap userid
book_features_df=rating_popular_book.pivot_table(values='Book-Rating', index=['Book-Title'],columns='User-ID', fill_value=0)
book_features_df.head()

# COMMAND ----------

# DBTITLE 1,Membuat model KNN
from scipy.sparse import csr_matrix

book_features_df_matrix = csr_matrix(book_features_df.values)

from sklearn.neighbors import NearestNeighbors


model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(book_features_df_matrix)

# COMMAND ----------

book_features_df.shape

# COMMAND ----------

# DBTITLE 1,Menghitung distance data
import numpy as np
#Mengambil nilai acar
query_index = np.random.choice(book_features_df.shape[0])
print(query_index)

#menghitung distance menggunakan model yang telah dibuat
distances, indices = model_knn.kneighbors(book_features_df.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)

# COMMAND ----------

# DBTITLE 1,Menampilkan Hasil Rekomendasi
for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(book_features_df.index.to_numpy()[query_index]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, book_features_df.index.to_numpy()[indices.flatten()[i]], distances.flatten()[i]))
