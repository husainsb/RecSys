# Load packages
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import avg, col, lit, expr, row_number, collect_list, size, array_contains, monotonically_increasing_id, udf, explode, min as spark_min, max as spark_max,countDistinct, when, isnan
from pyspark.sql.window import Window
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
from pyspark.sql.types import FloatType, DoubleType
from pyspark.ml.linalg import SparseVector, Vectors
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from sklearn.preprocessing import normalize
import faiss
from collections import defaultdict

# Read data and initiate Spark session
# Initialize Spark
spark = SparkSession.builder.appName("HybridRecommender").getOrCreate()

# Load Ratings and Movies Data into Pandas, then convert to Spark DataFrames
ratings_pd = pd.read_csv('./sample_data/ratings.dat', sep='::', names=['user_id', 'movie_id', 'rating', 'timestamp'], engine='python')
movies_pd = pd.read_csv('./sample_data/movies.dat', sep='::', names=['movie_id', 'title', 'genres'], engine='python',encoding='latin-1')

# Remove special chars, else words will be broken into sub-words
movies_pd['genres'] = movies_pd['genres'].str.replace('Sci-Fi','SciFi')
movies_pd['genres'] = movies_pd['genres'].str.replace('Film-Noir','Noir')

ratings = spark.createDataFrame(ratings_pd[['user_id', 'movie_id', 'rating']])
movies = spark.createDataFrame(movies_pd)

# Split ratings into train and test
train_ratings, test_ratings = ratings.randomSplit([0.8,0.2])

# Collaborative Filtering RS

# Cross Validation on ALS using Grid Search
als = ALS(userCol='user_id',itemCol='movie_id',ratingCol='rating',coldStartStrategy="drop",implicitPrefs=False,
          seed=32,maxIter=20)


# Define parameters to search on
paramGrid = ParamGridBuilder() \
    .addGrid(als.regParam, [1, 0.1, 0.01,0.05]) \
    .addGrid(als.rank, [10, 20,30]) \
    .build()

# Create CV
crossval = CrossValidator(estimator=als,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(predictionCol="prediction", labelCol="rating", metricName="mae"),
                          numFolds=3,parallelism=4)

# Fit CV on training set of Ratings
cv_model = crossval.fit(train_ratings)

# Extract best performing parameters
best_rank = cv_model.bestModel._java_obj.parent().getRank()
best_regParam = cv_model.bestModel._java_obj.parent().getRegParam()
best_model_params = {'rank': best_rank, 'regParam': best_regParam}

print(f"Best rank: {best_rank}")  # 10
print(f"Best regParam: {best_regParam}")  # 0.5

print(cv_model.avgMetrics) #avg MAE across all model params combo

# train best ALS
als = ALS(userCol='user_id',itemCol='movie_id',ratingCol='rating', \
          rank=best_model_params.get('rank'), \
          regParam=best_model_params.get('regParam'), \
          coldStartStrategy="drop",
          implicitPrefs=False)

model_cf = als.fit(train_ratings)

# Calculate 100 recommendations for all users at one go
user_recs = model_cf.recommendForAllUsers(100)
user_recs = user_recs.selectExpr("user_id", "explode(recommendations) as rec")
user_recs = user_recs.selectExpr("user_id", "rec.movie_id as movie_id", "rec.rating as cf_score")

# user_recs.groupBy("user_id").count().show(5)

# Content Based Filtering (CBF) RS
# Using Spark ML TF-IDF vectorization technique on movie Genre and storing it into FAISS HNSW vector store

# Spark TF-IDF on Genres
movies = movies.withColumn("genres", expr("translate(genres, '|', ' ')"))
tokenizer = Tokenizer(inputCol="genres", outputCol="words")
words_data = tokenizer.transform(movies)

hashing_tf = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurized_data = hashing_tf.transform(words_data)

idf = IDF(inputCol="rawFeatures", outputCol="features")
idf_model = idf.fit(featurized_data)
rescaled_data = idf_model.transform(featurized_data)

# FAISS Index for scalable content-based retrieval
# Convert Spark TF-IDF features to NumPy

movie_features_pd = rescaled_data.select("movie_id", "features").rdd.map(lambda row: (row[0], row[1].toArray())).collect()
movie_ids, features = zip(*movie_features_pd)
features = np.vstack(features).astype('float32')

#Normalize features for cosine similarity
features_norm = normalize(features, axis=1, norm='l2').astype('float32')

# Create FAISS HNSW index using dot product (cosine similarity)
dimension = features_norm.shape[1]
index = faiss.IndexHNSWFlat(dimension, 32,faiss.METRIC_INNER_PRODUCT)  # 32 = #neighbors in HNSW
index.hnsw.efConstruction = 40
index.add(features_norm)

# Search top-k similar movies for each item
k_similar = 100
# Recommend similar movies using FAISS for new movies
faiss_neighbors = {}
# Compute average cosine similarity per movie (excluding self)
cbf_score_map = defaultdict(float)
for i, vec in enumerate(features_norm):
    D, I = index.search(np.array([vec]).astype('float32'), k_similar)
    faiss_neighbors[movie_ids[i]] = [movie_ids[j] for j in I[0] if movie_ids[j] != movie_ids[i]]
    sim_sum = 0.0
    count = 0
    for j, dist in zip(I[0], D[0]):
        if movie_ids[i] != movie_ids[j]:
            sim_sum += dist
            count += 1
    #avg cosine-similiarities for each movie
    if count > 0:
        cbf_score_map[movie_ids[i]] = sim_sum / count
    else:
      cbf_score_map[movie_ids[i]]=0.0

# Convert to Spark DataFrame
cbf_scores_pd = pd.DataFrame(list(cbf_score_map.items()), columns=["movie_id", "cbf_score"])
cbf_scores_sdf = spark.createDataFrame(cbf_scores_pd)

# cbf_scores_pd['cbf_score'].describe()

# add guard-rails so score can Infinities/nulls can be converted to 0
cbf_scores_sdf = cbf_scores_sdf.withColumn(
    "cbf_score", when(col("cbf_score").isNull() |
                      isnan(col("cbf_score")) |
                      (col("cbf_score") == float("inf")) |
                      (col("cbf_score") == float("-inf")) |
                      (col("cbf_score") < float(-5e5)), 0.0).otherwise(col("cbf_score")))

# check Min and Max Cosine scores - should be between 0-1
cbf_scores_sdf.select(spark_min("cbf_score"),spark_max("cbf_score")).show(truncate=False)

# Check for similiar movies by picking any random movie

movie_to_chk=3091
print(movies_pd[movies_pd['movie_id']==movie_to_chk])

movies_pd[movies_pd['movie_id'].isin(faiss_neighbors.get(movie_to_chk))] #genres are very similar to chosen movie

# Create Hybrid RecSys
# Weights are set to 0.5 for both CF and CBF in Hybrid score

# Join CF and CBF scores
hybrid_scores = user_recs.join(cbf_scores_sdf, on="movie_id", how="left")

# Handle missing values in CBF scores if any
hybrid_scores = hybrid_scores.fillna({"cbf_score": 0.0})

# Normalize CF scores to [0, 1] so it can be added with CBF
cf_stats = hybrid_scores.select(spark_min("cf_score").alias("cf_min"), spark_max("cf_score").alias("cf_max")).first()

cf_min, cf_max = cf_stats["cf_min"], cf_stats["cf_max"]

hybrid_scores = hybrid_scores.withColumn("cf_score_norm", (col("cf_score") - lit(cf_min)) / (lit(cf_max) - lit(cf_min)))
hybrid_scores = hybrid_scores.withColumn("hybrid_score", expr("0.5 * cf_score_norm + 0.5 * cbf_score"))

# Evaluation on all 3 models
# Join entire Ratings dataset for evaluation by keeping high Rated movies only
relevant_set = ratings.filter(col("rating") > 3.0).groupBy("user_id").agg(collect_list("movie_id").alias("relevant"))

# Evaluate precision@k and ndcg@k using Spark SQL array functions
from pyspark.sql.functions import expr, collect_list, explode, size, array_intersect, array_position, array, sort_array

def evaluate_recommender_variant(score_column):
  # Generate recommendations
  topk = 10
  window = Window.partitionBy("user_id").orderBy(col(score_column).desc())
  top_recs = hybrid_scores.withColumn("rank", row_number().over(window)).filter(col("rank") <= topk).groupBy("user_id").agg(collect_list("movie_id").alias("recommended"))

  # Join with relevance info
  eval_df = top_recs.join(relevant_set, on="user_id", how="inner")
  eval_df = eval_df.withColumn("intersection", array_intersect("recommended", "relevant"))
  eval_df = eval_df.withColumn("precision", size("intersection") / lit(topk))

  eval_df = eval_df.withColumn("ordered_relevant", sort_array("relevant"))
  eval_df = eval_df.withColumn("dcg", expr("aggregate(sequence(0, size(recommended)-1), 0D, (acc, i) -> acc + IF(array_contains(relevant, recommended[i]), 1 / log2(i+2), 0))"))
  eval_df = eval_df.withColumn("idcg", expr("aggregate(sequence(0, least(size(relevant), {0})-1), 0D, (acc, i) -> acc + 1 / log2(i+2))".format(topk)))
  eval_df = eval_df.withColumn("ndcg", expr("IF(idcg > 0, dcg / idcg, 0)"))

  # Average metrics
  results = eval_df.select("user_id", "precision", "ndcg")
  results.cache()

  metrics = results.agg({"precision": "avg", "ndcg": "avg"}).first()
  avg_precision = metrics["avg(precision)"]
  avg_ndcg = metrics["avg(ndcg)"]

  return avg_precision, avg_ndcg

metrics_cf = evaluate_recommender_variant("cf_score_norm")
metrics_cbf = evaluate_recommender_variant("cbf_score")
metrics_hybrid = evaluate_recommender_variant("hybrid_score")

# Print Comparison
print("\n--- Recommendation Quality Comparison ---")
print(f"Model \t\t| {'Precision@k':<12} | {'NDCG@k'}")
print(f"Collabarative | {metrics_cf[0]:<12.4f} | {metrics_cf[1]:.4f}")
print(f"Content Based | {metrics_cbf[0]:<12.4f} | {metrics_cbf[1]:.4f}")
print(f"Hybrid \t\t| {metrics_hybrid[0]:<12.4f} | {metrics_hybrid[1]:.4f}")

# Cold start strategies
# For new users: Recommend top popular movies

topk=10
popular_movies = train_ratings.groupBy("movie_id").count().orderBy(col("count").desc()).limit(topk)
popular_movies.show()

# For new items: Recommend to users with similar past rated genres

# Step 1: Build user genre profiles
train_with_genres = train_ratings.join(movies.select("movie_id", "genres"), on="movie_id")
user_genre_profile = train_with_genres.groupBy("user_id", "genres").count()

# Step 2: Recommend new items to users who rated similar genres in past
new_items = movies.join(train_ratings, on="movie_id", how="left_anti")  # unseen items
user_genre_profile = user_genre_profile.withColumnRenamed("genres", "user_genres")
new_item_candidates = new_items.join(user_genre_profile, new_items.genres == user_genre_profile.user_genres)

recommended_new_items = new_item_candidates.groupBy("user_id").agg(collect_list("movie_id").alias("new_recommendations"))
recommended_new_items.show(truncate=False)

# Show New movies
new_items.show()

spark.stop()
