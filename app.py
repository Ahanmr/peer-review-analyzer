import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import umap.umap_ as umap
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns
from collections import Counter
import re
from collections import defaultdict
from html import escape
from datetime import datetime
import nltk
import statsmodels.api as sm

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Add custom CSS
def local_css():
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: 600;
            color: #1E88E5;
            padding: 1rem 0;
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: 500;
            color: #424242;
            padding: 0.5rem 0;
        }
        .stat-box {
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            margin: 0.5rem 0;
        }
        .stat-number {
            font-size: 2rem;
            font-weight: 600;
            color: #1E88E5;
        }
        .stat-label {
            font-size: 1rem;
            color: #666;
        }
        .sidebar-header {
            font-size: 1.5rem;
            font-weight: 500;
            color: #1E88E5;
            padding: 1rem 0;
        }
        .section-divider {
            margin: 2rem 0;
            border-top: 1px solid #dee2e6;
        }
        .highlight-box {
            background-color: #e3f2fd;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

# Page config
st.set_page_config(page_title="OpenReview Analysis", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    # Read the full dataset
    df = pd.read_csv('data/reviews.csv')
    
    # Get unique paper_ids to ensure we sample complete sets of reviews
    unique_papers = df['paper_id'].unique()
    
    # If we have more than 100 rows, sample the data
    if len(df) > 100:
        # Randomly sample paper_ids
        sampled_papers = np.random.choice(unique_papers, size=min(len(unique_papers), 20), replace=False)
        # Filter for the sampled papers
        df = df[df['paper_id'].isin(sampled_papers)]
        # If still more than 100 rows, take a random sample
        if len(df) > 100:
            df = df.sample(n=100, random_state=42)
    
    # Convert review_date to datetime if present
    if 'review_date' in df.columns:
        df['review_date'] = pd.to_datetime(df['review_date'], format='%d %b %Y', errors='coerce')
    
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
def generate_wordcloud(texts, custom_stopwords=None):
    # Default stopwords
    default_stopwords = set(['author', 'paper', 'review', 'reviewer', 'model', 
                           'network', 'book', 'research', 'work', 'method'])
    
    if custom_stopwords:
        default_stopwords.update(custom_stopwords)
    
    # Create WordCloud with custom stopwords
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=default_stopwords,
        min_font_size=10,
        max_font_size=50,
        colormap='viridis'
    )
    
    # Simple word frequency approach instead of TextBlob
    text = ' '.join(str(t) for t in texts)
    return wordcloud.generate(text)

# Add this function for keyword highlighting and scoring
def highlight_keyword(text, keyword, window_size=50):
    # Case-insensitive search
    pattern = re.compile(f'({keyword})', re.IGNORECASE)
    
    # Find all matches and their positions
    matches = list(pattern.finditer(text))
    
    if not matches:
        return text, []
    
    # Get context windows around matches
    contexts = []
    for match in matches:
        start = max(0, match.start() - window_size)
        end = min(len(text), match.end() + window_size)
        
        # Get context and highlight keyword
        context = text[start:end]
        highlighted = pattern.sub(r'<span style="background-color: yellow">\1</span>', context)
        
        # Calculate a simple relevance score based on position and surrounding words
        position_score = 1 - (match.start() / len(text))  # Earlier mentions score higher
        contexts.append({
            'context': highlighted,
            'position_score': position_score,
            'surrounding_words': len(context.split()),
        })
    
    return text, contexts

# Main app
def main():
    local_css()
    
    # Load data at the start of main()
    df = load_data()
    
    # Sidebar styling
    st.sidebar.markdown('<div class="sidebar-header">Navigation</div>', unsafe_allow_html=True)
    
    # Add timestamp and dataset info to sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Embeddings", "Review Analysis", "Word Clouds", "Reviewer Insights", "Keyword Search"],
        format_func=lambda x: f"📊 {x}" if x == "Overview" 
                    else f"🔍 {x}" if x == "Keyword Search"
                    else f"📈 {x}" if x == "Review Analysis"
                    else f"☁️ {x}" if x == "Word Clouds"
                    else f"👥 {x}" if x == "Reviewer Insights"
                    else f"🎯 {x}"
    )
    
    # Main content
    if page == "Overview":
        st.markdown('<div class="main-header">OpenReview Analysis Dashboard</div>', unsafe_allow_html=True)
        
        # Summary statistics in a grid
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                f"""
                <div class="stat-box">
                    <div class="stat-number">{len(df)}</div>
                    <div class="stat-label">Total Reviews</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div class="stat-box">
                    <div class="stat-number">{df['paper_id'].nunique()}</div>
                    <div class="stat-label">Unique Papers</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with col3:
            avg_confidence = df['confidence'].mean() if 'confidence' in df.columns else 'N/A'
            st.markdown(
                f"""
                <div class="stat-box">
                    <div class="stat-number">{avg_confidence:.2f}</div>
                    <div class="stat-label">Average Confidence</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Distribution plots
        st.markdown('<div class="sub-header">Score Distributions</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'recommendation' in df.columns:
                fig = px.histogram(
                    df, 
                    x='recommendation', 
                    title='Distribution of Recommendations',
                    color_discrete_sequence=['#1E88E5']
                )
                fig.update_layout(
                    template='plotly_white',
                    margin=dict(t=30, l=10, r=10, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'confidence' in df.columns:
                fig = px.histogram(
                    df, 
                    x='confidence', 
                    title='Distribution of Reviewer Confidence',
                    color_discrete_sequence=['#43A047']
                )
                fig.update_layout(
                    template='plotly_white',
                    margin=dict(t=30, l=10, r=10, b=10)
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Quick insights
        st.markdown('<div class="sub-header">Quick Insights</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                """
                <div class="highlight-box">
                    <h4>📈 Top Statistics</h4>
                    <ul>
                        <li>Average review length: {} words</li>
                        <li>Most reviewed paper: {} reviews</li>
                    </ul>
                </div>
                """.format(
                    int(df['review_comments'].str.split().str.len().mean()) if 'review_comments' in df.columns else 'N/A',
                    df.groupby('paper_id').size().max()
                ),
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                """
                <div class="highlight-box">
                    <h4>🎯 Key Metrics</h4>
                    <ul>
                        <li>Acceptance rate: {}%</li>
                        <li>Average recommendation: {:.2f}</li>
                    </ul>
                </div>
                """.format(
                    int(df['paper_accepted'].mean() * 100) if 'paper_accepted' in df.columns else 'N/A',
                    df['recommendation'].mean() if 'recommendation' in df.columns else 'N/A'
                ),
                unsafe_allow_html=True
            )
    
    elif page == "Embeddings":
        st.markdown('<div class="main-header">Embeddings Visualization</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            embed_source = st.selectbox(
                "Choose text source",
                ["paper_title", "paper_abstract", "review_comments"]
            )
        
        with col2:
            dim_reduction = st.selectbox(
                "Dimensionality reduction",
                ["t-SNE", "UMAP"]
            )
        
        with col3:
            color_by = st.selectbox(
                "Color by",
                ["sentiment", "recommendation", "confidence"]
            )
        
        if embed_source in df.columns:
            valid_texts = df[embed_source].dropna()
            if len(valid_texts) > 0:
                embeddings = generate_embeddings(valid_texts)
                reduced = reduce_dimensions(embeddings, 'tsne' if dim_reduction == 't-SNE' else 'umap')
                
                # Create DataFrame for plotting with proper indexing
                plot_df = pd.DataFrame(
                    reduced,
                    columns=['x', 'y'],
                    index=valid_texts.index
                )
                plot_df['text'] = valid_texts
                if color_by in df.columns:
                    plot_df['color'] = df.loc[valid_texts.index, color_by]
                else:
                    plot_df['color'] = 0  # default color if column not available
                
                fig = px.scatter(
                    plot_df,
                    x='x',
                    y='y',
                    color='color',
                    title=f"{embed_source} Embeddings using {dim_reduction}",
                    hover_data=['text'],
                    color_continuous_scale='viridis'
                )
                
                fig.update_layout(
                    template='plotly_white',
                    width=800,
                    height=600
                )
                
                st.plotly_chart(fig)
                
                # Add clustering analysis
                st.subheader("Cluster Analysis")
                from sklearn.cluster import KMeans
                
                n_clusters = st.slider("Number of clusters", 2, 10, 5)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(reduced)
                
                plot_df['cluster'] = clusters
                fig = px.scatter(
                    plot_df,
                    x='x',
                    y='y',
                    color='cluster',
                    title=f"Cluster Analysis ({n_clusters} clusters)",
                    hover_data=['text']
                )
                st.plotly_chart(fig)
    
    elif page == "Review Analysis":
        st.markdown('<div class="main-header">Review Analysis</div>', unsafe_allow_html=True)
        
        # Add sentiment analysis with explanation
        if 'review_comments' in df.columns:
            df['sentiment'] = df['review_comments'].apply(analyze_sentiment)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    df,
                    x='sentiment',
                    title='Distribution of Review Sentiments',
                    color_discrete_sequence=['#2ecc71']
                )
                st.plotly_chart(fig)
                
                # Add explanation
                st.write("""
                **Sentiment Analysis Explanation:**
                - Negative values (-1 to 0) indicate negative sentiment
                - Positive values (0 to 1) indicate positive sentiment
                - The distribution shows the overall tone of reviews
                """)
            
            with col2:
                if 'recommendation' in df.columns:
                    fig = px.scatter(
                        df,
                        x='sentiment',
                        y='recommendation',
                        title='Sentiment vs Recommendation',
                        color='confidence'
                    )
                    st.plotly_chart(fig)
        
        # Add detailed sentiment breakdown
        st.subheader("Detailed Sentiment Analysis")
        
        # Sample positive and negative reviews
        positive_sample = df[df['sentiment'] > 0.5].sample(min(3, len(df)))
        negative_sample = df[df['sentiment'] < -0.5].sample(min(3, len(df)))
        
        st.write("**Sample Positive Reviews:**")
        for _, row in positive_sample.iterrows():
            st.write(f"- Sentiment Score: {row['sentiment']:.2f}")
            st.write(f"- Review: {row['review_comments'][:200]}...")
        
        st.write("**Sample Negative Reviews:**")
        for _, row in negative_sample.iterrows():
            st.write(f"- Sentiment Score: {row['sentiment']:.2f}")
            st.write(f"- Review: {row['review_comments'][:200]}...")
    
    elif page == "Word Clouds":
        st.markdown('<div class="main-header">Word Clouds</div>', unsafe_allow_html=True)
        
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
        st.markdown('<div class="main-header">Reviewer Insights</div>', unsafe_allow_html=True)
        
        # Calculate review statistics for columns that exist
        stats_columns = {}
        for col in ['confidence', 'recommendation', 'impact', 'originality', 'clarity']:
            if col in df.columns:
                stats_columns[f'Avg {col.capitalize()}'] = df.groupby('paper_id')[col].mean()
        
        stats_columns['Review Count'] = df.groupby('paper_id').size()
        review_stats = pd.DataFrame(stats_columns)
        
        # Correlation matrix
        corr_matrix = review_stats.corr()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Review Score Correlations")
            fig = px.imshow(
                corr_matrix,
                color_continuous_scale='RdBu',
                title="Correlation Matrix of Review Scores"
            )
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Score Distributions")
            available_metrics = [col for col in ['confidence', 'impact', 'originality', 'clarity'] 
                               if col in df.columns]
            if available_metrics:
                selected_metric = st.selectbox("Select metric", available_metrics)
                
                fig = px.box(
                    df,
                    y=selected_metric,
                    title=f"Distribution of {selected_metric.capitalize()} Scores"
                )
                st.plotly_chart(fig)
        
        # Reviewer behavior analysis
        st.subheader("Reviewer Behavior Analysis")
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Create scatter plot only with available columns
            if all(col in df.columns for col in ['confidence', 'recommendation']):
                plot_data = df.copy()
                
                # Handle missing values for size and color parameters
                size_col = 'impact' if 'impact' in df.columns else 'confidence'
                color_col = 'clarity' if 'clarity' in df.columns else 'confidence'
                
                # Fill NaN values with mean for size parameter
                plot_data[size_col] = plot_data[size_col].fillna(plot_data[size_col].mean())
                
                fig = px.scatter(
                    plot_data,
                    x='confidence',
                    y='recommendation',
                    size=size_col,
                    color=color_col,
                    title="Multi-dimensional Score Analysis",
                    labels={
                        'confidence': 'Confidence',
                        'recommendation': 'Recommendation',
                        size_col: size_col.capitalize(),
                        color_col: color_col.capitalize()
                    }
                )
                st.plotly_chart(fig)
        
        with col4:
            # Time-based analysis if dates are available
            if 'review_date' in df.columns:
                time_metrics = [col for col in ['confidence', 'recommendation'] 
                              if col in df.columns]
                if time_metrics:
                    fig = px.line(
                        df.groupby('review_date')[time_metrics].mean(),
                        title="Score Trends Over Time"
                    )
                    st.plotly_chart(fig)
            
            # If no time data, show alternative visualization
            else:
                if 'recommendation' in df.columns:
                    fig = px.histogram(
                        df,
                        x='recommendation',
                        title="Distribution of Recommendations",
                        nbins=20
                    )
                    st.plotly_chart(fig)
    
    elif page == "Keyword Search":
        st.markdown('<div class="main-header">Keyword Search</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            keyword = st.text_input("Enter keyword to search in reviews:")
        with col2:
            min_score = st.slider("Minimum relevance score", 0.0, 1.0, 0.3)
        
        search_button = st.button("Search")
        
        if search_button and keyword:
            if 'review_comments' in df.columns:
                matches = df[df['review_comments'].str.contains(keyword, case=False, na=False)]
                st.write(f"Found {len(matches)} reviews containing '{keyword}'")
                
                results = []
                for _, row in matches.iterrows():
                    text, contexts = highlight_keyword(row['review_comments'], keyword)
                    if contexts:
                        # Calculate overall relevance score
                        avg_score = np.mean([c['position_score'] for c in contexts])
                        if avg_score >= min_score:
                            results.append({
                                'title': row['paper_title'],
                                'text': text,
                                'contexts': contexts,
                                'score': avg_score,
                                'confidence': row.get('confidence', 'N/A'),
                                'recommendation': row.get('recommendation', 'N/A')
                            })
                
                # Sort results by relevance score
                results.sort(key=lambda x: x['score'], reverse=True)
                
                for result in results:
                    st.write("---")
                    st.write(f"**Paper Title:** {result['title']}")
                    st.write(f"**Relevance Score:** {result['score']:.2f}")
                    st.write(f"**Reviewer Confidence:** {result['confidence']}")
                    st.write(f"**Recommendation:** {result['recommendation']}")
                    
                    # Display highlighted contexts
                    st.write("**Matching Contexts:**")
                    for ctx in result['contexts']:
                        st.markdown(f"...{ctx['context']}...", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
