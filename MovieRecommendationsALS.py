from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import lit
from typing import Dict

def load_movie_names() -> Dict[int, str]:
    """
    Load movie names from the u.item file.
    
    Returns:
        A dictionary mapping movie IDs to movie names.
    """
    movie_names = {}
    with open("ml-100k/u.item", encoding="ISO-8859-1") as f:
        for line in f:
            fields = line.split('|')
            movie_names[int(fields[0])] = fields[1]
    return movie_names

def parse_input(line: Row) -> Row:
    """
    Parse each line of the input file.
    
    Args:
        line: A Row object containing the raw input line.
    
    Returns:
        A Row object containing userID, movieID, and rating.
    """
    fields = line.value.split()
    return Row(userID=int(fields[0]), movieID=int(fields[1]), rating=float(fields[2]))

def main():
    spark = SparkSession.builder.appName("MovieRecommendations").getOrCreate()
    spark.conf.set("spark.sql.crossJoin.enabled", "true")

    movie_names = load_movie_names()

    lines = spark.read.text("hdfs:///user/maria_dev/ml-100k/u.data").rdd
    ratings_rdd = lines.map(parse_input)
    ratings = spark.createDataFrame(ratings_rdd).cache()

    als = ALS(maxIter=5, regParam=0.01, userCol="userID", itemCol="movieID", ratingCol="rating")
    model = als.fit(ratings)

    print("\nRatings for user ID 0:")
    user_ratings = ratings.filter("userID = 0")
    for rating in user_ratings.collect():
        print(f"{movie_names[rating['movieID']]}: {rating['rating']}")

    print("\nTop 20 recommendations:")
    rating_counts = ratings.groupBy("movieID").count().filter("count > 100")
    popular_movies = rating_counts.select("movieID").withColumn('userID', lit(0))

    recommendations = model.transform(popular_movies)
    top_recommendations = recommendations.sort(recommendations.prediction.desc()).take(20)

    for recommendation in top_recommendations:
        print(f"{movie_names[recommendation['movieID']]}: {recommendation['prediction']:.2f}")

    spark.stop()

if __name__ == "__main__":
    main()