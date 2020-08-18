import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def load_movs_shows(path_one, path_two):
    """
    Return combined dataset between two given files, one with movies and one
    with tv shows info, keeps all rows and columns from both datasets
    missing values are NaN
    """
    movies = pd.read_csv(path_one)
    shows = pd.read_csv(path_two)
    # concat stacks the two datasets, no duplicate column names
    movs_shows = pd.concat([movies, shows], axis=0, sort=True)
    return movs_shows


def load_imdb(path_one, path_two):
    """
    Return the joined dataset between two given files, one with imdb and
    one with movie info returned dataset only has rows where corresponding
    movie titles match in the given two datasets
    Resulting dataset has information of movies present in both given datasets
    """
    imdb = pd.read_csv(path_one)
    movies = pd.read_csv(path_two)
    merged = movies.merge(imdb, left_on='Title', right_on='title', how='inner')
    return merged


def netflix_analysis(df):
    netflix = df[df['Netflix'] == 1]  
    new = netflix['Genres'].str.split(",", expand = True)
    col1 = new[0]
    col2 = new[1]
    col3 = new[2]
    col4 = new[3]
    col5 = new[4]
    col6 = new[5]
    col7 = new[6]
    col8 = new[7]
    temp = col1.append(col2)
    temp2 = temp.append(col4)
    temp3 = temp2.append(col5)
    temp4 = temp3.append(col6)
    temp5 = temp4.append(col7)
    temp6 = temp5.append(col8)
    temp6 = temp6.dropna()
    result = temp6.unique()
    length = len(result)
    print(result)
    print(length)


def hulu_analysis(df):
    hulu = df[df['Hulu'] == 1]
    new = hulu['Genres'].str.split(",", expand = True)
    col1 = new[0]
    col2 = new[1]
    col3 = new[2]
    col4 = new[3]
    col5 = new[4]
    col6 = new[5]
    col7 = new[6]
    temp = col1.append(col2)
    temp2 = temp.append(col4)
    temp3 = temp2.append(col5)
    temp4 = temp3.append(col6)
    temp5 = temp4.append(col7)
    temp5 = temp5.dropna()
    result = temp5.unique()
    length = len(result)
    print(result)
    print(length)


def disney_analysis(df):
    disney = df[df['Disney+'] == 1]
    new = disney['Genres'].str.split(",", expand = True)
    col1 = new[0]
    col2 = new[1]
    col3 = new[2]
    col4 = new[3]
    col5 = new[4]
    col6 = new[5]
    col7 = new[6]
    col8 = new[7]
    temp = col1.append(col2)
    temp2 = temp.append(col4)
    temp3 = temp2.append(col5)
    temp4 = temp3.append(col6)
    temp5 = temp4.append(col7)
    temp6 = temp5.append(col8)
    temp6 = temp6.dropna()
    result = temp6.unique()
    length = len(result)
    print(result)
    print(length)
    


def prime_analysis(df):
    prime = df[df['Prime Video'] == 1]
    new = prime['Genres'].str.split(",", expand = True)
    col1 = new[0]
    col2 = new[1]
    col3 = new[2]
    col4 = new[3]
    col5 = new[4]
    col6 = new[5]
    col7 = new[6]
    col8 = new[7]
    col9 = new[8]
    temp = col1.append(col2)
    temp2 = temp.append(col4)
    temp3 = temp2.append(col5)
    temp4 = temp3.append(col6)
    temp5 = temp4.append(col7)
    temp6 = temp5.append(col8)
    temp7 = temp6.append(col9)
    temp7 = temp7.dropna()
    result = temp7.unique()
    length = len(result)
    print(result)
    print(length)

def main():
    movs_shows = load_movs_shows('data/Movies_Streaming_Platforms.csv',
                                 'data/TvShows_Streaming_Platforms.csv')
    df = pd.read_csv('data/Movies_Streaming_Platforms.csv')
    netflix_analysis(df)
    hulu_analysis(df)
    disney_analysis(df)
    prime_analysis(df)


if __name__ == '__main__':
    main()
