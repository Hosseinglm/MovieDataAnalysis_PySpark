from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql import functions as F
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

def parse_input(line: str) -> Row:
    """
    Parse each line of the input file.
    
    Args:
        line: A string containing user ID, movie ID, rating, and timestamp.
    
    Returns:
        A Row object containing movieID and rating.
    """
    fields = line.split()
    return Row(movieID=int(fields[1]), rating=float(fields[2]))

def main():
    # Create a SparkSession
    spark = SparkSession.builder.appName("LowestRatedPopularMovies").getOrCreate()

    # Load movie names
    movie_names = load_movie_names()

    # Load and process the u.data file
    lines = spark.sparkContext.textFile("hdfs:///user/maria_dev/ml-100k/u.data")
    movies = lines.map(parse_input)
    movie_dataset = spark.createDataFrame(movies)

    # Compute average rating and count for each movieID
    averages_and_counts = movie_dataset.groupBy("movieID").agg(
        F.avg("rating").alias("avg_rating"),
        F.count("rating").alias("count")
    )

    # Filter movies rated more than 10 times and sort by average rating
    popular_movies = averages_and_counts.filter("count > 10").orderBy("avg_rating")

    # Take the bottom 10 results
    bottom_ten = popular_movies.limit(10).collect()

    # Print results
    print("The 10 lowest-rated movies (with more than 10 ratings):")
    for movie in bottom_ten:
        print(f"{movie_names[movie['movieID']]}: {movie['avg_rating']:.2f} ({movie['count']} ratings)")

    spark.stop()

if __name__ == "__main__":
    main()