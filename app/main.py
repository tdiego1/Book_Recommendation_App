# Streamlit library's
import numpy as np
import pandas
import pandas as pd
import scipy.sparse
import sklearn.neighbors
import streamlit as st
from utils import read_data, head, body

# Streamlit header
head()



# Filter out warnings
import warnings
warnings.filterwarnings("ignore")

# Read in CSV files
books = read_data('/Users/dtorres/PycharmProjects/Book_Recommendation_App/data/Books.csv')
users = read_data('/Users/dtorres/PycharmProjects/Book_Recommendation_App/data/Users.csv')
ratings = read_data('/Users/dtorres/PycharmProjects/Book_Recommendation_App/data/Ratings.csv')

# Remove unnecessary columns
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-M']]

# Rename the columns for each data set
books.rename(columns= {'Book-Title':'title', 'Book-Author':'author', 'Year-Of-Publication': 'year', 'Publisher': 'publisher',
                       'Image-URL-M': 'image'}, inplace=True)
users.rename(columns= {'User-ID':'user_id', 'Location':'location', 'Age':'age'}, inplace=True)
ratings.rename(columns= {'User-ID':'user_id', 'Book-Rating':'rating'}, inplace=True)

# Get users who have 150 reviews or more
x = ratings['user_id'].value_counts() > 200
y = x[x].index
ratings = ratings[ratings['user_id'].isin(y)]

# Merge ratings with books
rating_with_book = ratings.merge(books, on='ISBN')

# Extract books that have received more than 50 ratings
number_rating = rating_with_book.groupby('title')['rating'].count().reset_index()
number_rating.rename(columns= {'rating':'number_of_ratings'}, inplace=True)
final_rating = rating_with_book.merge(number_rating, on='title')
final_rating = final_rating[final_rating['number_of_ratings'] >= 100]
final_rating.drop_duplicates(['user_id', 'title'], inplace=True)

# Create pivot table
book_pivot = final_rating.pivot_table(columns='user_id', index='title', values='rating')
book_pivot.fillna(0, inplace=True)

# Create matrix
book_sparse = scipy.sparse.csr_matrix(book_pivot)

# Train nearest neighbor algorithm
model = sklearn.neighbors.NearestNeighbors(algorithm='brute')
model.fit(book_sparse)

# Gets input from user.
selection = st.selectbox('Select a movie:', book_pivot.index)
sel_index = book_pivot.index.get_loc(selection)

# Check if there is a book selected.
if selection is not None:
    # Find image URL and display book
    image = books.loc[books['title'] == selection, 'image'].iloc[0]
    st.image(image, caption=selection, use_column_width='never')

# Get suggestions
distances, suggestions = model.kneighbors(book_pivot.iloc[sel_index, :].values.reshape(1, -1))

for i in range(len(suggestions)):
    print(book_pivot.index[suggestions[i]])




