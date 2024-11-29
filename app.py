import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import umap
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns
from collections import Counter

# Page config
st.set_page_config(page_title="OpenReview Analysis", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('reviews.csv')
    # Convert review_date to datetime if present
    if 'review_date' in df.columns:
        df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    return df

# Generate embeddings
@st.cache_data
def generate_embeddings(texts, method='tfidf'):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    embeddings = vectorizer.fit_transform(texts).toarray()
    return embeddings

# Dimensionality reduction
@st.cache_data
def reduce_dimensions(embeddings, method='tsne'):
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:  # umap
        reducer = umap.UMAP(random_state=42)
    reduced = reducer.fit_transform(embeddings)
    return reduced

# Sentiment analysis
@st.cache_data
def analyze_sentiment(text):
    return TextBlob(str(text)).sentiment.polarity

# Generate word cloud
def generate_wordcloud(texts):
    text = ' '.join(texts)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    return wordcloud

# Main app
def main():
    st.title("OpenReview Analysis Dashboard")
    
    # Load data
    df = load_data()
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Embeddings", "Review Analysis", "Word Clouds", "Reviewer Insights", "Keyword Search"]
    )
    
    if page == "Overview":
        st.header("Dataset Overview")
        st.write(f"Total number of reviews: {len(df)}")
        st.write(f"Number of unique papers: {df['paper_id'].nunique()}")
        
        # Basic statistics
        if 'recommendation' in df.columns:
            fig = px.histogram(df, x='recommendation', title='Distribution of Recommendations')
            st.plotly_chart(fig)
            
        if 'confidence' in df.columns:
            fig = px.histogram(df, x='confidence', title='Distribution of Reviewer Confidence')
            st.plotly_chart(fig)
    
    elif page == "Embeddings":
        st.header("Embeddings Visualization")
        
        col1, col2 = st.columns(2)
        with col1:
            embed_source = st.selectbox(
                "Choose text source for embeddings",
                ["paper_title", "paper_abstract", "review_comments"]
            )
        
        with col2:
            dim_reduction = st.selectbox(
                "Choose dimensionality reduction method",
                ["t-SNE", "UMAP"]
            )
        
        # Generate and display embeddings
        if embed_source in df.columns:
            valid_texts = df[embed_source].dropna()
            if len(valid_texts) > 0:
                embeddings = generate_embeddings(valid_texts)
                reduced = reduce_dimensions(embeddings, 'tsne' if dim_reduction == 't-SNE' else 'umap')
                
                fig = px.scatter(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    title=f"{embed_source} Embeddings using {dim_reduction}"
                )
                st.plotly_chart(fig)
    
    elif page == "Review Analysis":
        st.header("Review Analysis")
        
        # Add sentiment analysis
        if 'review_comments' in df.columns:
            df['sentiment'] = df['review_comments'].apply(analyze_sentiment)
            
            fig = px.histogram(df, x='sentiment', title='Distribution of Review Sentiments')
            st.plotly_chart(fig)
            
            # Show correlation between sentiment and recommendation if available
            if 'recommendation' in df.columns:
                fig = px.scatter(df, x='sentiment', y='recommendation', 
                               title='Sentiment vs Recommendation')
                st.plotly_chart(fig)
    
    elif page == "Word Clouds":
        st.header("Word Clouds")
        
        if 'review_comments' in df.columns and 'recommendation' in df.columns:
            col1, col2 = st.columns(2)
            
            # Positive reviews
            with col1:
                st.subheader("Positive Reviews")
                positive_reviews = df[df['recommendation'] >= df['recommendation'].median()]['review_comments']
                wordcloud = generate_wordcloud(positive_reviews)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud)
                plt.axis('off')
                st.pyplot(plt)
            
            # Negative reviews
            with col2:
                st.subheader("Negative Reviews")
                negative_reviews = df[df['recommendation'] < df['recommendation'].median()]['review_comments']
                wordcloud = generate_wordcloud(negative_reviews)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud)
                plt.axis('off')
                st.pyplot(plt)
    
    elif page == "Reviewer Insights":
        st.header("Reviewer Insights")
        
        if 'confidence' in df.columns and 'recommendation' in df.columns:
            fig = px.scatter(df, x='confidence', y='recommendation',
                           title='Confidence vs Recommendation')
            st.plotly_chart(fig)
    
    elif page == "Keyword Search":
        st.header("Keyword Search")
        
        keyword = st.text_input("Enter keyword to search in reviews:")
        search_button = st.button("Search")
        
        if search_button and keyword:
            if 'review_comments' in df.columns:
                matches = df[df['review_comments'].str.contains(keyword, case=False, na=False)]
                st.write(f"Found {len(matches)} reviews containing '{keyword}'")
                
                for _, row in matches.iterrows():
                    st.write("---")
                    st.write(f"**Paper Title:** {row['paper_title']}")
                    st.write(f"**Review:** {row['review_comments']}")

if __name__ == "__main__":
    main()
