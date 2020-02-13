import pandas as pd
import numpy as np
from scipy import spatial
from scipy.sparse import csr_matrix

# obtain dataset from two files
rating_data = pd.read_csv('/Users/shaomengyuan/Desktop/movie-lens-small-latest-dataset/ratings.csv')
movie_data = pd.read_csv('/Users/shaomengyuan/Desktop/movie-lens-small-latest-dataset/movies.csv')

# merge two files to generate a dataset which is what I want.
dataset = pd.merge(rating_data, movie_data, on='movieId')
dataset.head()
# sort the dataset by different index.
dataset.groupby('title')['rating'].mean().head()
dataset.groupby('title')['rating'].mean().sort_values(ascending = False).head()
dataset.groupby('title')['rating'].count().sort_values(ascending = False).head()

# build a dataframe contains title, rating and counts.
rating_mean = pd.DataFrame(dataset.groupby('title')['rating'].mean())
rating_mean['counts'] = pd.DataFrame(dataset.groupby('title')['rating'].count())
rating_mean.head()
# print(rating_mean.head(20))

# users-item-matrix uses userId as index, titile as columes and fill with rating.
users_items_matrix = dataset.pivot_table(index = 'userId', columns = 'title', values='rating')
users_items_matrix.head()

# movie_similarity used to get a list which sort by correlation.
def movie_similarity(movie_title):
    # get a movie form matrix and know which user rating it.
    movie_ratings = users_items_matrix[movie_title]
    movie_ratings.head()
    # correlation
    movie_similar_it = users_items_matrix.corrwith(movie_ratings)
    # build a dataframe contains title, Correlation and counts, sort them by Correlation,
    # only the counts of rating greater than 50 can be showed, and also the first 8.
    corr_with_movie = pd.DataFrame(movie_similar_it, columns= ['Correlation'])
    corr_with_movie.dropna(inplace= True)
    corr_with_movie.head()
    corr_with_movie.sort_values('Correlation', ascending= False).head()
    corr_with_movie = corr_with_movie.join(rating_mean['counts'])
    corr_with_movie.head()
    similar_list = corr_with_movie[corr_with_movie['counts']>50].sort_values('Correlation', ascending = False).head(8)
    # a = similar_list.values[1][1]
    # similar_list = np.array(similar_list)
    return similar_list
# movie_similarity('Toy Story (1995)')

# use userid to extract watched list of the user.
def user_list(userid):
    # create a new list and store movies name into.
    movies_List = []
    movies_name = np.array(users_items_matrix.columns.values.tolist())
    # create a matrix fill with 0 if it's NaN.
    users_and_items = dataset.pivot_table(index = 'userId', columns = 'title', values='rating').fillna(0)
    movies_names = users_and_items.columns.values.tolist()
    user_watched = users_and_items.values[userid-1]
    # get the watched list since rating value will be not 0.
    for number in range(0, len(user_watched)):
        if (users_and_items.iloc[userid-1][number] != 0):
            movies_List.append(movies_name[number])      
    return movies_List
# abc = user_list(2)

def movie_list(user_movies):
    # create a empty dataframe
    storage = pd.DataFrame()
    # fill it with the first 8 similarity movies to each movie.
    for movie in user_movies:
        storage = storage.append(movie_similarity(movie))
        # storage = pd.concat(storage, movie_similarity(movie))
    # sort them according to their correlation to movie and choose first 10 objects.
    storage = storage.sort_values('Correlation', ascending = False).head(10)
    print(storage)
    # storage = set(storage).head(10)
    # print(storage)
    return storage
# movie_list(abc)

def main():
    # ask input user's ID and convert it to Int.
    x = int(input('Please type UserID:'))
    # measure the value.
    if(x < 0 ):
        print('ID cannot less than 0')
        main()
    else:
        get_list = user_list(x)
        movie_list(get_list)

if __name__ == "__main__":
    main()