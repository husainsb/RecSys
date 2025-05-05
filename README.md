# RecSys
Scalable Recommendation system

## Overview

Here I am going to show, how to build a scalable Hybrid Recommendation system which combines Collaborative filtering and Content-based filtering to recommend top movies to users based on their history/preferences.

## Dataset

I have used MovieLens dataset of 1 Million user ratings on 3.8K movies by 6K users
https://grouplens.org/datasets/movielens/1m/
Kindly download data from above link before running the code.

## Collaborative Filtering

1. I have used Spark MLlib which can scale to millions of data points.
2. This allows for efficient training and recommendation generation across distributed environments.
3. It uses Spark ALS for Matrix Factorization – which is fast and efficient
4. Since ratings are given this is Explicit Preference type filtering
5. I have pre-computed the ALS Recommendations for all users. (100 recommendations per user)
6. Used Cross validation to find best performing params for ALS

## Content Based Filtering

1. Used the movie genre as main feature to find similar movie for the chosen movie.
2. Used Spark TF-IDF vectorizer to compute feature vector.
3. Used normalized vector and the Dot product to compute Cosine Similarity scores
4. These scores/vectors are stored in FAISS HNSW vector store for fast querying and searching

## Hybrid Recommendations

1. Collaborative Filtering and Content based Filtering scores are normalized (0-1) and then joined on Movie and added to get the final score
2. Weights of both scores are set to 0.5 to give equal weightage

## Evaluation Metrics
I have employed the evaluation metrics like NDCG@10 and Precision@10 for our 3 models: Collaborative Filtering, Content based Filtering and Hybrid RecSys.

## Cold Start strategy

1. For new users, the most popular movies can be recommended. Popularity based on movies which got highest views (being rated) from users.
2. For new movies, we can recommend to users those movies for which the genre is similar to those movies whom they have rated highest in past.

## Final notes

1. Content based Filtering can be improved if we employ Embeddings instead of TF-IDF vectorization features like Genre, Title, etc.
2. Collaborative filtering can be improved by eliminating one-timer users, improve Parameter search for Cross validation, etc
3. Eliminated use of pandas UDF which caused some issues like serialization/picklin
4. Using native Spark functionality wherever required as this makes evaluation faster and scalable.
5. NDCG and Precision scores are calculated using built-in Sparks SQL functions which makes computation very fast
6. Evaluation metrics can be improved if we eliminated one-time users and focus on improving the Collaborative Filtering and Content based filtering
7. The entire system was designed from scalable perspective, so more focus was on engineering side rather than accuracy of recommendations.
