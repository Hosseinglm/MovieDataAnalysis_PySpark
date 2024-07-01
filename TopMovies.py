from mrjob.job import MRJob
from mrjob.step import MRStep

class RatingsBreakdown(MRJob):
    def steps(self):
        return [
            MRStep(mapper=self.mapper_get_ratings,
                   reducer=self.reducer_count_ratings),
            MRStep(reducer=self.reducer_sorted_output)
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
        yield str(sum(values)).zfill(5), key

    def reducer_sorted_output(self, count, movies):
        """
        Output the results sorted by rating count.
        """
        for movie in movies:
            yield movie, count

if __name__ == '__main__':
    RatingsBreakdown.run()