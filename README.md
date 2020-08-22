# Analyzing Streaming Platforms

## Instructions
Simply run our analysis.py file to run our modules, produce our plots, and print our analysis. Run our test.py file to test modules with return statements on our small datasets. 
The datasets we used are linked below.
### Datasets
IMDb Movies Extensive Dataset
Link: https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset
Selection of Movies for Streaming Services
Link: https://www.kaggle.com/ruchi798/movies-on-netflix-prime-video-hulu-and-disney
Selection of TV Shows for Streaming Services
Link: https://www.kaggle.com/ruchi798/tv-shows-on-netflix-prime-video-hulu-and-disney

### Libraries used
pandas, sklearn, matplotlib.pyplot, statistics

### To better navigate our methods...
load_movs_shows: merges movies and tv shows datasets
load_imdb: merges imdb and movies datasets
count: prints returns of number of unique titles in movies and tv shows merged dataset, or any other dataset you input
avg_rating: returns overall average rating for given streaming platforms movies and dataset
genre_count: returns names and count of unique genre in given platform and dataset
predict_rating: machine learning method, returns tuple of predicted imdb rating mean squared error for training and test set
unique_list: returns unique genres from given dataset and specified platform
movs_per_genre: generates and saves plots for each streaming platform and includes plots like box chart for counts and percentages of movies in each genre for given sites as well as bar charts for genre distribution based on percentages as well