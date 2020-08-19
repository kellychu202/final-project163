import genre
import analysis
import pandas as pd
from cse163_utils import assert_equals

def test_genre():
    small_movies = pd.read_csv('data/small_movies.csv')
    assert_equals(5, genre.genre_count(small_movies, 'Netflix'))
    assert_equals(5, genre.genre_count(small_movies, 'Hulu'))
    assert_equals(5, genre.genre_count(small_movies, 'Disney+'))
    assert_equals(5, genre.genre_count(small_movies, 'Prime Video'))