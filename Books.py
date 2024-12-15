# Databricks notebook source
pip install pyspark matplotlib seaborn

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when, isnan
from pyspark import SparkFiles
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark.pandas as pd
import numpy as np
import pandas as pd

# COMMAND ----------

spark = SparkSession.builder \
    .appName("EDA_Books") \
    .getOrCreate()


# COMMAND ----------

spark

# COMMAND ----------

spark.sparkContext.addFile("https://yudiantosujana.com/files/books/books.csv")
spark.sparkContext.addFile("https://yudiantosujana.com/files/books/users.csv")
spark.sparkContext.addFile("https://yudiantosujana.com/files/books/ratings.csv")

# COMMAND ----------

books = spark.read.csv("file://"+SparkFiles.get('books.csv'), header=True, inferSchema=True)
ratings = spark.read.csv("file://"+SparkFiles.get('ratings.csv'), header=True, inferSchema=True)
users = spark.read.csv("file://"+SparkFiles.get('users.csv'), header=True, inferSchema=True)

# COMMAND ----------

books.show(5)

# COMMAND ----------

books.printSchema()
ratings.printSchema()
users.printSchema()

# COMMAND ----------

print(f"Books: {books.count()} rows, {len(books.columns)} columns")
print(f"Ratings: {ratings.count()} rows, {len(ratings.columns)} columns")
print(f"Users: {users.count()} rows, {len(users.columns)} columns")


# COMMAND ----------

for df, name in zip([books, ratings, users], ['Books', 'Ratings', 'Users']):
    print(f"{name} Missing Values:")
    df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns]).show()


# COMMAND ----------

# Filter usia valid
users_clean = users.filter((col('Age') >= 10) & (col('Age') <= 100))

# Tambahkan kolom rentang usia per 5 tahun
users_with_age_group = users_clean.withColumn(
    "Age-Group",
    when((col("Age") >= 0) & (col("Age") <= 5), "0-5")
    .when((col("Age") > 5) & (col("Age") <= 10), "6-10")
    .when((col("Age") > 10) & (col("Age") <= 15), "11-15")
    .when((col("Age") > 15) & (col("Age") <= 20), "16-20")
    .when((col("Age") > 20) & (col("Age") <= 25), "21-25")
    .when((col("Age") > 25) & (col("Age") <= 30), "26-30")
    .when((col("Age") > 30) & (col("Age") <= 35), "31-35")
    .when((col("Age") > 35) & (col("Age") <= 40), "36-40")
    .when((col("Age") > 40) & (col("Age") <= 45), "41-45")
    .when((col("Age") > 45) & (col("Age") <= 50), "46-50")
    .when((col("Age") > 50) & (col("Age") <= 55), "51-55")
    .when((col("Age") > 55) & (col("Age") <= 60), "56-60")
    .when((col("Age") > 60) & (col("Age") <= 65), "61-65")
    .when((col("Age") > 65) & (col("Age") <= 70), "66-70")
    .when((col("Age") > 70) & (col("Age") <= 75), "71-75")
    .when((col("Age") > 75) & (col("Age") <= 80), "76-80")
    .when((col("Age") > 80) & (col("Age") <= 85), "81-85")
    .when((col("Age") > 85) & (col("Age") <= 90), "86-90")
    .when((col("Age") > 90) & (col("Age") <= 95), "91-95")
    .when((col("Age") > 95) & (col("Age") <= 100), "96-100")
    .otherwise("Unknown")
)



# COMMAND ----------

age_group_distribution = users_with_age_group.groupBy("Age-Group").count().orderBy("Age-Group")
# Konversi ke Pandas DataFrame
age_group_distribution_pd = age_group_distribution.toPandas()
age_group_distribution.show()

# Visualisasi
sns.barplot(data=age_group_distribution_pd, x="Age-Group", y="count", palette="viridis")
plt.title("Distribusi Pengguna per Rentang Usia")
plt.xlabel("Rentang Usia (tahun)")
plt.ylabel("Jumlah Pengguna")
plt.xticks(rotation=45)
plt.show()


# COMMAND ----------

ratings_books = ratings.join(books, on="ISBN", how="inner")

# Hitung rata-rata rating
average_ratings = ratings_books.groupBy("Book-Title") \
    .agg(avg("Book-Rating").alias("Average-Rating")) \
    .orderBy(col("Average-Rating").desc())

average_ratings.show(20)


# COMMAND ----------

ratings_users = ratings.join(users, on="User-ID", how="inner")
ratings_books_users = ratings_users.join(books, on="ISBN", how="inner")

# Tambahkan kolom rentang usia
from pyspark.sql.functions import when
ratings_books_users = ratings_books_users.withColumn(
    "Age-Group", when(col("Age").between(10, 20), "10-20")
                .when(col("Age").between(21, 30), "21-30")
                .when(col("Age").between(31, 40), "31-40")
                .when(col("Age").between(41, 50), "41-50")
                .otherwise("51+")
)

# Hitung buku paling populer di setiap rentang usia
popular_books_by_age = ratings_books_users.groupBy("Age-Group", "Book-Title") \
    .agg(count("Book-Rating").alias("Count")) \
    .orderBy("Age-Group", col("Count").desc())



# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

# Tambahkan nomor baris berdasarkan Age-Group dan jumlah rating
window_spec = Window.partitionBy("Age-Group").orderBy(col("Count").desc())
top_books_by_age = popular_books_by_age.withColumn("Rank", row_number().over(window_spec))

# Ambil hanya buku peringkat 1 untuk setiap rentang usia
top_books_by_age = top_books_by_age.filter(col("Rank") == 1).drop("Rank")
top_books_by_age.show()


# COMMAND ----------

top_books_pd = top_books_by_age.toPandas()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.barplot(data=top_books_pd, x="Age-Group", y="Count", hue="Book-Title", dodge=False, palette="viridis")
plt.title("Buku Paling Populer Berdasarkan Rentang Usia")
plt.xlabel("Rentang Usia")
plt.ylabel("Jumlah Rating")
plt.xticks(rotation=45)
plt.legend(title="Judul Buku", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Plot grouped bar chart
plt.figure(figsize=(12, 6))
ax = sns.barplot(data=top_books_pd, x="Age-Group", y="Count", hue="Book-Title", palette="viridis")

# Tambahkan label jumlah rating di atas setiap batang
for bars in ax.containers:
    ax.bar_label(bars, fmt='%.0f', label_type='edge')  # fmt='%.0f' untuk angka bulat

# Tambahkan judul, label, dan legenda
plt.title("Top 3 Buku Paling Populer Berdasarkan Rentang Usia")
plt.xlabel("Rentang Usia")
plt.ylabel("Jumlah Rating")
plt.xticks(rotation=45)
plt.legend(title="Judul Buku", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# COMMAND ----------

from pyspark.sql.functions import col, avg, count

# Gabungkan dataset
ratings_users = ratings.join(users, on="User-ID", how="inner")
ratings_books_users = ratings_users.join(books, on="ISBN", how="inner")


# COMMAND ----------

# Hitung rata-rata rating dan jumlah rating untuk setiap buku
book_ratings_stats = ratings_books_users.groupBy("Book-Title") \
    .agg(
        avg("Book-Rating").alias("Average-Rating"),  # Rata-rata rating
        count("Book-Rating").alias("Number-of-Ratings")  # Jumlah rating
    )

book_ratings_stats.show()


# COMMAND ----------

# Urutkan buku berdasarkan Average-Rating dan Number-of-Ratings
best_books = book_ratings_stats.orderBy(col("Average-Rating").desc(), col("Number-of-Ratings").desc())

best_books.show(10)  # Menampilkan 10 buku teratas


# COMMAND ----------

# Ambil buku dengan rating terbaik dan jumlah rating terbanyak
best_book = best_books.limit(1)
best_book.show()


# COMMAND ----------

# Mengonversi hasil ke Pandas DataFrame
best_books_pd = best_books.toPandas()

# Tampilkan beberapa data untuk melihat formatnya
best_books_pd.head()


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

# Menampilkan 10 buku terbaik berdasarkan rating tertinggi dan jumlah rating terbanyak
plt.figure(figsize=(12, 8))
sns.barplot(x='Average-Rating', y='Book-Title', data=best_books_pd.head(10), palette="viridis")

# Tambahkan jumlah rating sebagai label di sebelah kanan batang
for index, value in enumerate(best_books_pd.head(10)['Number-of-Ratings']):
    plt.text(value + 0.05, index, str(value), color='black', va='center')

# Set label dan title
plt.title("Top 10 Buku dengan Rating Terbaik dan Jumlah Rating Terbanyak")
plt.xlabel("Rata-Rata Rating")
plt.ylabel("Judul Buku")
plt.tight_layout()

# Menampilkan grafik
plt.show()


# COMMAND ----------

# Filter data untuk buku "Wild Animus"
wild_animus_ratings = ratings_books_users.filter(ratings_books_users["Book-Title"] == "Wild Animus")

# Hitung jumlah rating dan rata-rata rating untuk buku "Wild Animus"
wild_animus_stats = wild_animus_ratings.agg(
    count("Book-Rating").alias("Number-of-Ratings"),
    avg("Book-Rating").alias("Average-Rating")
)

wild_animus_stats.show()


# COMMAND ----------

from pyspark.sql.functions import count, avg

# Hitung jumlah rating dan rata-rata rating untuk setiap buku
book_ratings_stats = ratings_books_users.groupBy("Book-Title") \
    .agg(
        count("Book-Rating").alias("Number-of-Ratings"),  # Jumlah rating
        avg("Book-Rating").alias("Average-Rating")  # Rata-rata rating
    )
# Urutkan buku berdasarkan jumlah rating terbanyak, kemudian berdasarkan rata-rata rating tertinggi
most_rated_books = book_ratings_stats.orderBy(col("Number-of-Ratings").desc(), col("Average-Rating").desc())

# Menampilkan beberapa buku teratas
most_rated_books.show(10)  # Menampilkan 10 buku teratas
# Ambil buku dengan rating terbanyak dan rata-rata rating terbaik
top_rated_book = most_rated_books.limit(1)
top_rated_book.show()


# COMMAND ----------

# MAGIC %md
# MAGIC Berdasarkan hasil EDA di atas, dapat diketahui fakta sebagai berikut
# MAGIC 1. Pengguna aktif terbanyak yaitu pengguna dengan rentang usia 26-30 tahun
# MAGIC 2. Buku yang populer di berbagai usia atau paling banyak mendapatkan rating adalah buku Wild Animus
# MAGIC 3. Pengguna cenderung memberikan rating pada buku yang tidak mereka sukai
