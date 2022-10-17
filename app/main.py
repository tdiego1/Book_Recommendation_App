import warnings
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import plotly.express as px
from utils import read_data, head

# Streamlit header
head()

# Filter out warnings

warnings.filterwarnings("ignore")


# ----READ IN CSV FILES----
books = read_data('/Users/dtorres/PycharmProjects/Book_Recommendation_App/data/Books.csv')
users = read_data('/Users/dtorres/PycharmProjects/Book_Recommendation_App/data/Users.csv')
ratings = read_data('/Users/dtorres/PycharmProjects/Book_Recommendation_App/data/Ratings.csv')


# ----PROCESS DATA----
# Remove columns that are not needed
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-M']]

# Rename the columns for each data set
books.rename(
    columns={'Book-Title': 'title', 'Book-Author': 'author', 'Year-Of-Publication': 'year', 'Publisher': 'publisher',
             'Image-URL-M': 'image'}, inplace=True)
users.rename(columns={'User-ID': 'user_id', 'Location': 'location', 'Age': 'age'}, inplace=True)
ratings.rename(columns={'User-ID': 'user_id', 'Book-Rating': 'rating'}, inplace=True)

# Get users who have 150 reviews or more
count_ratings = ratings['user_id'].value_counts() > 150
y = count_ratings[count_ratings].index
ratings = ratings[ratings['user_id'].isin(y)]

# Merge ratings with books
rating_with_book = ratings.merge(books, on='ISBN')

# Extract books that have received more than 50 ratings
num_rating = rating_with_book.groupby('title')['rating'].count().reset_index()
num_rating.rename(columns={'rating': 'number_of_ratings'}, inplace=True)
final_rating = rating_with_book.merge(num_rating, on='title')
final_rating = final_rating[final_rating['number_of_ratings'] >= 50]

# Remove duplicate records
final_rating.drop_duplicates(['user_id', 'title'], inplace=True)

# Create pivot table
book_pivot = final_rating.pivot_table(columns='user_id', index='title', values='rating')
book_pivot.fillna(0, inplace=True)


# ----MODEL DATA----
# Create matrix
book_sparse = scipy.sparse.csr_matrix(book_pivot)

# Model data with Cosine Similarity
model = cosine_similarity(book_sparse)


# ----GETS BOOK SELECTION FROM USER----
# Gets input from user.
selection = st.selectbox('Select a book:', book_pivot.index)
sel_index = book_pivot.index.get_loc(selection)

# Check if there is a book selected.
if selection is not None:
    st.markdown("##### This is your selected book:")

    # Create columns
    col1, col2, col3 = st.columns(3)

    # Find image URL and display book
    image = books.loc[books['title'] == selection, 'image'].iloc[0]

    # Centers Book image.
    with col1:
        st.write('')
    with col2:
        st.image(image, caption=selection, width=100)
    with col3:
        st.write('')


# ----GET RECOMMENDED BOOKS----
# Remove selected book from suggestion pool
book_pivot.drop(index=selection, inplace=True)

# Get suggestions
suggestions_list = list(enumerate(model[sel_index]))
suggestions = sorted(suggestions_list, key=lambda x: x[1], reverse=True)[1:7]


# ----DISPLAY RECOMMENDED BOOKS----
# Get image files and titles for book suggestions
rec_images = []
rec_titles = []
for i in range(len(suggestions)):
    rec_images.append(books.loc[books['title'] == book_pivot.index[int(suggestions[i][0])], 'image'].iloc[0])
    rec_titles.append(books.loc[books['title'] == book_pivot.index[int(suggestions[i][0])], 'title'].iloc[0])

# Display Recommended books
st.markdown("##### These are your recommended books:")
st.image(rec_images, caption=rec_titles, width=100)

# Display seperator
st.markdown("---")


# ----DISPLAY SCATTER PLOT----
# Get user ages older than 12 and less than 90
x_values = []
user_ages = users[~users.age.isnull()]
user_ages = user_ages[user_ages['age'] < 90]
user_ages = user_ages[user_ages['age'] > 12]
user_ages = user_ages[['user_id', 'age']]

# Merge ratings with user ages and average the ratings
ages_ratings = final_rating.groupby('user_id', as_index=False, sort=False)['rating'].mean()
ages_ratings = ages_ratings.merge(user_ages, on='user_id')
ages_ratings.rename(columns={'rating': 'avg_rating'}, inplace=True)

# Display scatter plot
plot = px.scatter(data_frame=ages_ratings, x='age', y='avg_rating',
                  labels={
                      'age': 'User Age',
                      'avg_rating': 'Average Rating'
                  },
                  title='Average Rating of User by Age')
st.plotly_chart(plot)


# ----DISPLAY HISTOGRAM----
# Group books by the same year together
year_reviews = rating_with_book.groupby('year')['rating'].count().reset_index()
year_reviews.year = pd.to_numeric(year_reviews.year, errors='coerce').fillna(0).astype('int')
year_reviews.rename(columns={'rating': 'count'}, inplace=True)

# Correct publishing year
year_reviews = year_reviews[(year_reviews['year'] >= 1900)]
year_reviews = year_reviews[year_reviews['year'] <= 2022]

# Display Histogram
plot2 = px.histogram(year_reviews, x='year', y='count', nbins=80, log_y=True,
                     labels={
                         'year': 'Publishing Year',
                         'count': 'Number of Ratings'
                     },
                     title='Number of Ratings by Publishing Year')
st.plotly_chart(plot2)


# ----DISPLAY BAR CHART----
# Get users and locations where the location is not Null
user_countries = users[~users.location.isnull()]
user_countries = user_countries[['user_id', 'location']]

# Get only countries
user_countries['location'] = user_countries['location'].str.rsplit(',').str[-1]
user_countries['location'] = user_countries['location'].str.strip()

# Replace empty values with NaN
user_countries['location'].replace('', np.nan, inplace=True)
user_countries['location'].replace('n/a', np.nan, inplace=True)
user_countries.dropna(subset=['location'], inplace=True)

# Change 'us' to 'usa'
user_countries['location'].replace('us', 'usa', inplace=True)

# Merge users and ratings
country_reviews = ratings.groupby('user_id', as_index=False, sort=False)['rating'].size()
country_reviews = country_reviews.merge(user_countries, on='user_id')
country_reviews.rename(columns={'size': 'count'}, inplace=True)
country_reviews = country_reviews.groupby('location', as_index=False, sort=False)['count'].sum()
country_reviews.sort_values(by=['count'], inplace=True, ascending=False)

# Display Bar Chart
plot3 = px.bar(country_reviews, x='location', y='count', log_y=True,
               labels={
                   'count': 'Number of Ratings',
                   'location': 'Country'
               },
               title='Number of Ratings Based on Country')
st.plotly_chart(plot3)
