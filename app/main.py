import warnings
import logging
import pandas as pd
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import plotly.express as px
from utils import read_data, head
from pathlib import Path

# Initialize logging settings
logging.basicConfig(filename='app.log', level='INFO', force=True)

# Streamlit header
head()

# Filter out warnings
warnings.filterwarnings("ignore")

# Get directories
home_dir = Path(__file__)
books_path = home_dir.parent.parent.joinpath('data/Books.csv')
user_path = home_dir.parent.parent.joinpath('data/Users.csv')
ratings_path = home_dir.parent.parent.joinpath('data/Ratings.csv')


# ----READ IN CSV FILES----
try:
    books = read_data(books_path)
    users = read_data(user_path)
    ratings = read_data(ratings_path)
except Exception as e:
    logging.exception(str(e))


# ----PROCESS DATA----
# Remove columns that are not needed
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-M']]

# Rename the columns for each data set
books.rename(
    columns={'Book-Title': 'title', 'Book-Author': 'author', 'Year-Of-Publication': 'year', 'Publisher': 'publisher',
             'Image-URL-M': 'image'}, inplace=True)
users.rename(columns={'User-ID': 'user_id', 'Location': 'location', 'Age': 'age'}, inplace=True)
ratings.rename(columns={'User-ID': 'user_id', 'Book-Rating': 'rating'}, inplace=True)


# ----DATA ANALYSIS----
# Remove users that do not have an age.
users = users[~users.age.isnull()]

# Get user ages older than 12 and less than 90
users = users[users['age'] > 12]
users = users[users['age'] < 90]

# Get only countries
users['location'] = users['location'].str.rsplit(',').str[-1]
users['location'] = users['location'].str.strip()

# Remove users that do not have a location
users = users[~users.location.isnull()]

# Replace empty values with NaN
#users['location'].replace('', np.nan, inplace=True)
#users['location'].replace('n/a', np.nan, inplace=True)
#users.dropna(subset=['location'], inplace=True)

# Change 'us' to 'usa'
users['location'].replace('us', 'usa', inplace=True)

# Get only ratings from the user list
ratings = ratings[ratings['user_id'].isin(users.index)]

# Get users who have 150 ratings or more
count_ratings = ratings['user_id'].value_counts() > 150
ratings_index = count_ratings[count_ratings].index
clean_ratings = ratings[ratings['user_id'].isin(ratings_index)]

# Merge ratings with books
ratings_with_book = clean_ratings.merge(books, on='ISBN')

# Convert year column into integers
ratings_with_book.year = pd.to_numeric(ratings_with_book.year, errors='coerce').fillna(0).astype('int')

# Remove ratings and books with a year less than or equal to 1900 and greater than or equal to  2022
ratings_with_book = ratings_with_book[(ratings_with_book['year'] >= 1900)]
ratings_with_book = ratings_with_book[ratings_with_book['year'] <= 2022]

# Extract books that have received more than 50 ratings
num_rating = ratings_with_book.groupby('title')['rating'].count().reset_index()
num_rating.rename(columns={'rating': 'number_of_ratings'}, inplace=True)
final_ratings = ratings_with_book.merge(num_rating, on='title')
final_ratings = final_ratings[final_ratings['number_of_ratings'] >= 50]

# Remove duplicate records
final_ratings.drop_duplicates(['user_id', 'title'], inplace=True)


# ----MODEL DATA----
try:
    # Create pivot table
    book_pivot = final_ratings.pivot_table(columns='user_id', index='title', values='rating')
    book_pivot.fillna(0, inplace=True)

    # Create matrix
    book_matrix = scipy.sparse.csr_matrix(book_pivot)

    # Model data with Cosine Similarity
    model = cosine_similarity(book_matrix)
except Exception as e:
    logging.error(str(e))


# ----GETS BOOK SELECTION FROM USER----
try:
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
except IOError as e:
    logging.error(str(e))


# ----GET RECOMMENDED BOOKS----
try:
    # Remove selected book from suggestion pool
    book_pivot.drop(index=selection, inplace=True)

    # Get suggestions
    suggestions_list = list(enumerate(model[sel_index]))
    suggestions = sorted(suggestions_list, key=lambda x: x[1], reverse=True)[1:6]
except Exception as e:
    logging.exception(str(e))

print(suggestions)


# ----DISPLAY RECOMMENDED BOOKS----
try:
    # Get image files and titles for book suggestions
    rec_images = []
    rec_titles = []
    for i in range(len(suggestions)):
        rec_images.append(books.loc[books['title'] == book_pivot.index[int(suggestions[i][0])], 'image'].iloc[0])
        rec_titles.append(books.loc[books['title'] == book_pivot.index[int(suggestions[i][0])], 'title'].iloc[0])

    # Display Recommended books
    st.markdown("##### These are your recommended books:")
    st.image(rec_images, caption=rec_titles, width=100)
except Exception as e:
    logging.exception(str(e))

# Display seperator
st.markdown("---")


# ----DISPLAY SCATTER PLOT----
# Get user ages older than 12 and less than 90
user_ages = users[['user_id', 'age']]

# Merge ratings with user ages and average the ratings
ages_ratings = clean_ratings.groupby('user_id', as_index=False, sort=False)['rating'].mean()
ages_ratings = ages_ratings.merge(user_ages, on='user_id')
ages_ratings.rename(columns={'rating': 'avg_rating'}, inplace=True)

# Display scatter plot
plot = px.scatter(data_frame=ages_ratings, x='age', y='avg_rating', trendline='ols',
                  labels={
                      'age': 'User Age',
                      'avg_rating': 'Avg Ratings'
                  },
                  title='Average Rating of User by Age')
st.plotly_chart(plot)


# ----DISPLAY HISTOGRAM----
# Group books by the same year together
year_reviews = ratings_with_book.groupby('year')['rating'].count().reset_index()
year_reviews.rename(columns={'rating': 'count'}, inplace=True)

# Display Histogram
plot2 = px.histogram(year_reviews, x='year', y='count', nbins=20, log_y=True,
                     labels={
                         'year': 'Publishing Year',
                         'count': 'Number of Ratings'
                     },
                     title='Number of Ratings by Publishing Year')
st.plotly_chart(plot2)


# ----DISPLAY BAR CHART----
# Get users and locations where the location is not Null
user_countries = users[['user_id', 'location']]

# Merge users and ratings
country_reviews = clean_ratings.groupby('user_id', as_index=False, sort=False)['rating'].size()
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
