from mrjob.job import MRJob
from mrjob.step import MRStep

class MostPopularMovie(MRJob):
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_ratings,
                   reducer=self.reducer_count_ratings),
            MRStep(reducer=self.reducer_find_max)
        ]

    def mapper_get_ratings(self, _, line):
        """
        Map each line to a (movieID, 1) pair.
        """
        (userID, movieID, rating, timestamp) = line.split('\t')
        yield movieID, 1

    def reducer_count_ratings(self, key, values):
        """
        Reduce by summing the ratings for each movie.
        """
        yield None, (sum(values), key)

    def reducer_find_max(self, key, values):
        """
        Find the movie with the maximum number of ratings.
        """
        yield max(values)

if __name__ == '__main__':
    MostPopularMovie.run()