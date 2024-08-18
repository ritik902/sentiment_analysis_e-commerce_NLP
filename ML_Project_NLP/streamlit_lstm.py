# write streamlit_lstm.py 
# run on cmd - streamlit run streamlit_lstm.py

import streamlit as st
# st.title('Sentiment Analysis Dashboard')

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model from the SavedModel directory
lstm_model = tf.keras.models.load_model('ML_Project_NLP/lstm_model.keras')
# Load the tokenizer used during training
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = joblib.load(handle)

# Load data
df = pd.read_csv('flipkart_streamlit.csv')

st.header('Real-Time Sentiment Prediction')
user_input = st.text_input('Enter a review:', '')

if user_input:
    # Preprocess input and make prediction
    sequences = tokenizer.texts_to_sequences([user_input])
    if len(sequences[0]) == 0:
        st.write("Input text is too short or not recognized. Please enter a longer review.")
    else:
        max_sequence_length = 100  
        user_input_padded = pad_sequences(sequences, maxlen=max_sequence_length)
        
        prediction = lstm_model.predict(user_input_padded)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        
        st.write(f'Sentiment: {sentiment}')


# ------------------------------------------------------------------------------------------------------------------


st.header('Sentiment distribution of flipkart data')

# WORD CLOUD
st.markdown('<h4>Word Cloud of positive and negative reviews', unsafe_allow_html=True)
# Separate positive and negative reviews
positive_reviews = df[df['Rate'] == 1]['Review'].str.cat(sep=' ')
negative_reviews = df[df['Rate'] == 0]['Review'].str.cat(sep=' ')

# Generate word clouds
wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_reviews)
wordcloud_negative = WordCloud(width=800, height=400, background_color='black').generate(negative_reviews)

# Plotting the word clouds
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Positive reviews word cloud
axs[0].imshow(wordcloud_positive, interpolation='bilinear')
axs[0].set_title("Positive Reviews Word Cloud", fontsize=16)
axs[0].axis('off')

# Negative reviews word cloud
axs[1].imshow(wordcloud_negative, interpolation='bilinear')
axs[1].set_title("Negative Reviews Word Cloud", fontsize=16)
axs[1].axis('off')

st.pyplot(fig)


# ------------------------------------------------------------------------------------------------------------------


# BAR and PIE PLOTS 
st.markdown('<h4>Bar plot and Pie chart', unsafe_allow_html=True)
count_rate = df['Rate'].value_counts()
fig, values = plt.subplots(1, 2, figsize=(10, 5))
# Bar chart
values[0].bar(count_rate.index, count_rate.values, color=['lightgreen', 'orange'])
values[0].set_xlabel('Review Sentiment')
values[0].set_ylabel('Number of Reviews')
values[0].set_title('Number of Positive and Negative Reviews')
values[0].set_xticks([0, 1])
values[0].set_xticklabels(['Negative', 'Positive'])

# Pie chart
values[1].pie(count_rate, labels=['Positive', 'Negative'], autopct='%1.1f%%', colors=['skyblue', 'yellow'])
values[1].set_title('Proportion of Positive and Negative Reviews')

# Adjust layout
plt.tight_layout()
st.pyplot(fig)


# ------------------------------------------------------------------------------------------------------------------


# TOP/BOTTOM PRODUCTS
st.markdown('<h4>Top 10 and Bottom 10 products according to ratings', unsafe_allow_html=True)
# Group the reviews by CleanedProductName and Rate to count the number of positive and negative reviews for each product
product_sentiment_counts = df.groupby(['CleanedProductName', 'Rate']).size().unstack(fill_value=0)
# Renamed the columns for clarity
product_sentiment_counts.columns = ['Negative', 'Positive']

# Sort the products by the number of positive reviews to find the top 10 products
top_10_products = product_sentiment_counts.sort_values(by='Positive', ascending=False).head(10)

# Sort the products by the number of negative reviews to find the bottom 10 products
bottom_10_products = product_sentiment_counts.sort_values(by='Negative', ascending=False).head(10)

# Plot the top 10 products
fig1, ax1 = plt.subplots(figsize=(10, 6))
top_10_products.plot(kind='bar', stacked=True, color=['red', 'green'], ax=ax1)
ax1.set_title('Top 10 Products with the Most Positive Reviews')
ax1.set_xlabel('Product Name')
ax1.set_ylabel('Number of Reviews')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
ax1.legend(title='Sentiment')
plt.tight_layout()

# Display the first plot in Streamlit
st.pyplot(fig1)

# Plot the bottom 10 products
fig2, ax2 = plt.subplots(figsize=(10, 6))
bottom_10_products.plot(kind='bar', stacked=True, color=['red', 'green'], ax=ax2)
ax2.set_title('Bottom 10 Products with the Most Negative Reviews')
ax2.set_xlabel('Product Name')
ax2.set_ylabel('Number of Reviews')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
ax2.legend(title='Sentiment')
plt.tight_layout()

# Display the second plot in Streamlit
st.pyplot(fig2)