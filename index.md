### How to Build A Movie Recomendation System 

### Data
Files:
1. TMDB Movie Dataset
- `https://www.kaggle.com/tmdb/tmdb-movie-metadata?select=tmdb_5000_credits.csv`
2. The Movies Dataset
- `https://www.kaggle.com/rounakbanik/the-movies-dataset?select=ratings.csv`
3. The Netflix Prize Dataset
- `https://www.kaggle.com/netflix-inc/netflix-prize-data`
3. File Name
- `file url`

### Outputs
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
### Code

```python
# process command line arguments
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
#process and load data 
import pandas as pd 
import numpy as np 

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


#merging the datasets based off 'id' 

df_credits.columns=['id','tittle','cast','crew']
df_movies=df_movies.merge(df_credits,on='id')

movie_id_list=df_credits['id'].tolist()
movie_title_list=df_credits['tittle'].tolist()

if args.print_extra_shit:
	print('Mergerd Data:')
	print(df_movies.head(5))
	print('\n')
```
```python
#initialize vars for imdb ranking system

C=df_movies['vote_average'].mean()
m=df_movies['vote_count'].quantile(.9)

if args.print_extra_shit:
	print('C:',C)
	print('m:',m)
	print('\n')
	
q_movies=df_movies.copy().loc[df_movies['vote_count'] >= m]

if args.print_extra_shit:
	print('q_movies.shape:',q_movies.shape)
	print('\n')
	
#weighted rating function:

def weighted_rating(x, m=m, C=C):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v/(v+m) * R) + (m/(m+v) * C)


# Define a new feature 'score' and calculate its value with `weighted_rating()`
q_movies['score'] = q_movies.apply(weighted_rating, axis=1)

#Sort movies based on score calculated above
q_movies = q_movies.sort_values('score', ascending=False)

#Print the top 15 movies
if args.print_data_info:
	print('Top 10 movies from new metric:')
	print(q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10))
	print('\n')

#show current popular movies
pop= df_movies.sort_values('popularity', ascending=False)

if args.print_data_info:
	print('Top 10 movies based off popularity:')
	print(pop[['title', 'vote_count', 'vote_average', 'popularity']].head(10))
	print('\n')
```
```python
# Build a term frequency inverse document frequency feature 

#Import TfIdfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

#Replace NaN with an empty string
df_movies['overview'] = df_movies['overview'].fillna('')

#Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(df_movies['overview'])

#Output the shape of tfidf_matrix
if args.print_extra_shit:
	print('tfidf_matrix.shape:')
	print(tfidf_matrix.shape)
	print('\n')

# Import linear_kernel
from sklearn.metrics.pairwise import linear_kernel

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Construct a reverse map of indices and movie titles
indices = pd.Series(df_movies.index, index=df_movies['title']).drop_duplicates()

# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return df_movies['title'].iloc[movie_indices]

if args.print_data_info:
	print('Example Recomendations for The Dark Knight Rises and the Avengers:')
	print('\n')
	print(get_recommendations('The Dark Knight Rises'))
	print('\n')
	print(get_recommendations('The Avengers'))
	print('\n')
```
```python
# Parse the stringified features into their corresponding python objects
from ast import literal_eval

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    df_movies[feature] = df_movies[feature].apply(literal_eval)

# Get the director's name from the crew feature. If director is not listed, return NaN
def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []

# Define new director, cast, genres and keywords features that are in a suitable form.
df_movies['director'] = df_movies['crew'].apply(get_director)
features = ['cast', 'keywords', 'genres']

for feature in features:
    df_movies[feature] = df_movies[feature].apply(get_list)


# Print the new features of the first 3 films
if args.print_extra_shit:
	print('new features of first 3 films:')
	print(df_movies[['title', 'cast', 'director', 'keywords', 'genres']].head(3))
	print('\n')
```
```python
# Function to convert all strings to lower case and strip names of spaces
def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        #Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# Apply clean_data function to your features.
features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    df_movies[feature] = df_movies[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])
df_movies['soup'] = df_movies.apply(create_soup, axis=1)

# Import CountVectorizer and create the count matrix
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(df_movies['soup'])


# Compute the Cosine Similarity matrix based on the count_matrix
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

# Reset index of our main DataFrame and construct reverse mapping as before
df_movies = df_movies.reset_index()
indices = pd.Series(df_movies.index, index=df_movies['title'])

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







## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/benfig1127/Data_Mining_Final_Project/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/benfig1127/Data_Mining_Final_Project/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
