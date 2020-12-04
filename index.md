# Building A Movie Recomendation System 



## Dataset Files
1. TMDB Movie Dataset
- `https://www.kaggle.com/tmdb/tmdb-movie-metadata?select=tmdb_5000_credits.csv`
2. The Movies Dataset
- `https://www.kaggle.com/rounakbanik/the-movies-dataset?select=ratings.csv`
3. The Netflix Prize Dataset
- `https://www.kaggle.com/netflix-inc/netflix-prize-data`


## Outputs
```
#Example Outputs:
Example Recomendations for The Dark Knight Rises and the Avengers:


		The Dark Knight
1		Batman Forever
2		Batman Returns
3		Batman
4		Batman: The Dark Knight Returns, Part 2
5		Batman Begins
6		Slow Burn
7		Batman v Superman: Dawn of Justice
8		JFK
9		Batman & Robin

                Avengers: Age of Ultron
1		Plastic
2		Timecop
3		This Thing of Ours
4		Thank You for Smoking
5		The Corruptor
6		Wall Street: Money Never Sleeps
7		Team America: World Police
8		The Fountain
9		Snowpiercer

```

```

Example Recomendations with new features added:


		The Dark Knight
1		Batman Begins
2		Amidst the Devil's Wings
3		The Prestige
4		Romeo Is Bleeding
5		Black November
6		Takers
7		Faster
8		Catwoman
9               Gangster Squad


		Avengers: Age of Ultron
1		Captain America: Civil War
2		Iron Man 2
3		Captain America: The First Avenger
4		The Incredible Hulk
5		Captain America: The Winter Soldier
6		Iron Man 3
7		X-Men: The Last Stand
8		Iron Man
9		Guardians of the Galaxy
```

## Code
```python
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--data_path_1', default='/home/bfigueroa20/data_mining/project_work/archive/tmdb_5000_credits.csv')
parser.add_argument('--data_path_2', default='/home/bfigueroa20/data_mining/project_work/archive/tmdb_5000_movies.csv')
parser.add_argument('--data_path_3', default='/home/bfigueroa20/data_mining/project_work/movies_dataset/ratings_small.csv')

parser.add_argument('--run_user1_test_case',action='store_true',default=False)
parser.add_argument('--num_movies',type=int)

parser.add_argument('--print_extra_shit',action='store_true',default=False)
parser.add_argument('--print_data_info',action='store_true',default=False)
args = parser.parse_args()
```

```python
import pandas as pd 
import numpy as np 

```
Loading and reading our data:

```python
print('\n')
print('Loading and computing movie metadata...')
print('\n')

df_credits=pd.read_csv(args.data_path_1)
df_movies=pd.read_csv(args.data_path_2)

if args.print_extra_shit:
	print('Raw Data:')
	print(df_credits.head(n=10))
	print(df_movies.head(n=10))
	print('\n')

df_credits.columns=['id','tittle','cast','crew']
df_movies=df_movies.merge(df_credits,on='id')

movie_id_list=df_credits['id'].tolist()
movie_title_list=df_credits['tittle'].tolist()

if args.print_extra_shit:
	print('Mergerd Data:')
	print(df_movies.head(5))
	print('\n')
```
Calculating the mean rating of all the movies in our dataset:
```python

C=df_movies['vote_average'].mean()

```
To appear on our list, a specific movie title must have more votes than 90% of the movies on the dataset
```python
m=df_movies['vote_count'].quantile(.9)
```
```python
if args.print_extra_shit:
	print('C:',C)
	print('m:',m)
	print('\n')
	
q_movies=df_movies.copy().loc[df_movies['vote_count'] >= m]

if args.print_extra_shit:
	print('q_movies.shape:',q_movies.shape)
	print('\n')
```
Next, we need to make sure to add another reference point besides just rating. As it stands, a movie with one favorable vote could appear ahead of a movie with hundreds of votes.

The formula below will allow us to factor this in: it places weights on both the average rating given and the number of ratings given, ensuring that the movies on our list are both widely watched and liked by those who did:

```python

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)

```

So far, we have created a simple recommender that takes in two factors, weighted according to our preference, and returns an ordered list. Now we will add other factors to make our recommender much more powerful.

### Content-Based Filtering

One thing we can analyze is the plot overview. If we can extract certain key words from the overview, we can match them up with other movies with similar key words.

To do this, we will calculate Term Frequency-Inverse Document Frequency (TF-IDF) vectors for each plot description.

A little background: Term Frequency refers to the number of times a word appears in a document. Inverse Document Frequency is the “relative” count of documents that contain the specific word and is defined as follows: log (# of documents/documents with specific term).

```python
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

q_movies = q_movies.sort_values('score', ascending=False)
```
Prints the top 10 movies
```python
if args.print_data_info:
	print('Top 10 movies from new metric:')
	print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10))
	print('\n')

pop= df_movies.sort_values('popularity', ascending=False)

if args.print_data_info:
	print('Top 10 movies based off popularity:')
	print(pop[['title', 'vote_count', 'vote_average', 'popularity']].head(10))
	print('\n')
```
We import scikit-learn, a machine learning library for Python, that can create the TF-IDF matrix for us:
```python
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

df_movies['overview'] = df_movies['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(df_movies['overview'])

if args.print_extra_shit:
	print('tfidf_matrix.shape:')
	print(tfidf_matrix.shape)
	print('\n')
```
To describe the “similarity” between two movies, we will use the **cosine similarity score**. This can be calculated by taking the dot product of our TF-IDF vectorizer: 
```python
from sklearn.metrics.pairwise import linear_kernel

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
```
We now want to define a function that, when given a movie title as input, will give us a list with 10 similar movies. To do this, we need a tool to locate the index of a movie in our metadata DataFrame, given the title: 
```python

indices = pd.Series(df_movies.index, index=df_movies['title']).drop_duplicates()

```
```python

```
Now that we have done that, we can properly define our recommender function. Here’s what we want: 
- Find the index of the movie from its title 
- Compute a list of cosine similarity scores for the given movie with all the movies in the dataset
- Grab the top ten most “similar” movies 

```python
def get_recommendations(title, cosine_sim=cosine_sim):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:11]

    movie_indices = [i[0] for i in sim_scores]

    return df_movies['title'].iloc[movie_indices]
```
```python
if args.print_data_info:
	print('Example Recomendations for The Dark Knight Rises and the Avengers:')
	print('\n')
	print(get_recommendations('The Dark Knight Rises'))
	print('\n')
	print(get_recommendations('The Avengers'))
	print('\n')
```
We have successfully created a recommender that can spit back similar movies to a given movie title based on the plot description. However, the accuracy of the recommender is not ideal. Feeding our recommender “The Dark Night Rises,” for example, results in a string of Batman Movies. We want a recommender that can recommend movies beyond a simple name association…

### Cast-Crew Recommender

To increase the accuracy of our recommender, we will be analyzing individuals involved in the movie as well its genre. We will use the director of the movie and the three top actors. To extract these datapoints, we will turn our data into a usable structure: 
```python
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df_movies[feature] = df_movies[feature].apply(literal_eval)
    
```
We then write functions to extract the key information: 
```python
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
```

```python
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    return []
```
```python
df_movies['director'] = df_movies['crew'].apply(get_director)
features = ['cast', 'keywords', 'genres']

for feature in features:
    df_movies[feature] = df_movies[feature].apply(get_list)
```
Printing it all out:
```python

if args.print_extra_shit:
	print('new features of first 3 films:')
	print(df_movies[['title', 'cast', 'director', 'keywords', 'genres']].head(3))
	print('\n')
```
To avoid the vectorizer confusing names, we turn all words to lowercase and remove all spaces: 
```python
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''
```
```python

features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    df_movies[feature] = df_movies[feature].apply(clean_data)
```
We can now take all of the appropriate metadata  and “feed it” into our vectorizer: 
```python
def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df_movies['soup'] = df_movies.apply(create_soup, axis=1)

```
These next steps are similar to how we constructed our Content-Based Filtering. The difference to note is that we utilize CountVectorizer as opposed to TF-IDF. The reason for this is that we do not want to negatively impact an actor’s presence in a film if they have acted in more movies than others. 
```python

from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df_movies['soup'])
```
```python

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

```
```python

df_movies = df_movies.reset_index()
indices = pd.Series(df_movies.index, index=df_movies['title'])
```
```python

if args.print_data_info:
	print('Example Recomendations with new features added:')
	print('\n')
	print(get_recommendations('The Dark Knight Rises',cosine_sim2))
	print('\n')
	print(get_recommendations('The Avengers',cosine_sim2))
	print('\n')
	
print('\n')	
print('Loaded movie metada, computing setup for user recomendation system...')
print('\n')
```

## Conclusion
This is a work-in-progress and we encourage all forms of feedback. Thank you for reading!
