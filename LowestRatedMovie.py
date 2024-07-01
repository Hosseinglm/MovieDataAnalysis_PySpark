from pyspark import SparkConf, SparkContext
from typing import Dict, Tuple

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

def parse_input(line: str) -> Tuple[int, Tuple[float, float]]:
    """
    Parse each line of the input file.
    
    Args:
        line: A string containing user ID, movie ID, rating, and timestamp.
    
    Returns:
        A tuple containing movie ID and a tuple of (rating, 1.0).
    """
    fields = line.split()
    return (int(fields[1]), (float(fields[2]), 1.0))

def main():
    conf = SparkConf().setAppName("LowestRatedMovies")
    sc = SparkContext(conf=conf)

    movie_names = load_movie_names()

    lines = sc.textFile("hdfs:///user/maria_dev/ml-100k/u.data")
    movie_ratings = lines.map(parse_input)

    rating_totals_and_count = movie_ratings.reduceByKey(lambda m1, m2: (m1[0] + m2[0], m1[1] + m2[1]))

    average_ratings = rating_totals_and_count.mapValues(lambda total_and_count: total_and_count[0] / total_and_count[1])

    sorted_movies = average_ratings.sortBy(lambda x: x[1])
    results = sorted_movies.take(10)

    print("The 10 lowest-rated movies:")
    for result in results:
        print(f"{movie_names[result[0]]}: {result[1]:.2f}")

    sc.stop()

if __name__ == "__main__":
    main()