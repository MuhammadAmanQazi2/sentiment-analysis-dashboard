# sentiment_analysis_dashboard.py
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Product Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Product Review Sentiment Analysis Dashboard")
st.markdown("Analyzing customer sentiment using Logistic Regression")

# Step 1: Data Loading and Preprocessing
@st.cache_data
def load_and_preprocess_data():
    # Load dataset
    df = pd.read_csv("synthetic_product_reviews.csv")
    
    # Convert Review Date to datetime and sort
    df['Review Date'] = pd.to_datetime(df['Review Date'])
    df = df.sort_values('Review Date')
    
    # Clean text data
    df['Review'] = df['Review'].str.lower().str.replace('[^\w\s]', '')
    
    # Map sentiment to numerical values
    df['sentiment_num'] = df['Sentiment'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})
    
    return df

df = load_and_preprocess_data()

# Step 2: Model Training
@st.cache_resource
def train_and_save_model():
    try:
        # Try to load existing model
        model = joblib.load('sentiment_model.pkl')
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        st.sidebar.success("Loaded pre-trained model")
    except:
        # Train new model if not found
        st.sidebar.info("Training new model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['Review'], df['sentiment_num'], test_size=0.2, random_state=42
        )
        
        # Create TF-IDF features
        tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        
        # Train model
        model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
        model.fit(X_train_tfidf, y_train)
        
        # Save model
        joblib.dump(model, 'sentiment_model.pkl')
        joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
        st.sidebar.success("Model trained and saved!")
    
    return model, tfidf

model, tfidf = train_and_save_model()

# Dashboard UI
# Sidebar for user inputs
with st.sidebar:
    st.header("Controls")
    show_raw_data = st.checkbox("Show raw data", value=False)
    min_rating = st.slider("Minimum rating", 1, 5, 1)
    selected_sentiment = st.multiselect(
        "Filter by sentiment",
        options=['Positive', 'Neutral', 'Negative'],
        default=['Positive', 'Neutral', 'Negative']
    )
    selected_category = st.multiselect(
        "Filter by category",
        options=df['Category'].unique(),
        default=df['Category'].unique()
    )

# Filter data based on selections
filtered_df = df[
    (df['Rating'] >= min_rating) & 
    (df['Sentiment'].isin(selected_sentiment)) &
    (df['Category'].isin(selected_category))
]

# Main dashboard layout
tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "⏳ Timeline", "🔍 Explore", "🤖 Predict"])

with tab1:
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews", len(df))
    col2.metric("Positive Rate", f"{len(df[df['Sentiment']=='Positive'])/len(df):.1%}")
    col3.metric("Avg Rating", f"{df['Rating'].mean():.1f} ⭐")
    col4.metric("Categories", len(df['Category'].unique()))
    
    # Sentiment distribution
    st.subheader("Sentiment Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Sentiment', order=['Positive', 'Neutral', 'Negative'], ax=ax)
    st.pyplot(fig)
    
    # Rating by category
    st.subheader("Average Rating by Category")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=df, x='Category', y='Rating', ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

with tab2:
    # Time series analysis
    st.subheader("Reviews Over Time")
    
    # Resample by time frequency
    time_freq = st.selectbox(
        "Time Frequency",
        options=['Weekly', 'Monthly', 'Quarterly'],
        index=1  # Default to Monthly
    )
    
    freq_map = {'Weekly': 'W', 'Monthly': 'M', 'Quarterly': 'Q'}
    ts_df = df.set_index('Review Date').resample(freq_map[time_freq]).agg({
        'Product Name': 'count',
        'Rating': 'mean',
        'Sentiment': lambda x: (x == 'Positive').mean()
    }).rename(columns={'Product Name': 'Review Count'})
    
    # Add smoothing
    ts_df['Positive_Ratio_Smoothed'] = ts_df['Sentiment'].rolling(
        window=3, min_periods=1, center=True
    ).mean()
    
    # Plot review count
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.bar(ts_df.index, ts_df['Review Count'], width=20, alpha=0.6)
    ax1.set_title(f"Number of Reviews ({time_freq})")
    ax1.set_ylabel("Review Count")
    st.pyplot(fig1)
    
    # Plot sentiment trend
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(ts_df.index, ts_df['Positive_Ratio_Smoothed'], 
            color='green', marker='o', linewidth=2)
    ax2.set_title(f"Positive Sentiment Ratio ({time_freq}, Smoothed)")
    ax2.set_ylabel("Positive Ratio")
    ax2.set_ylim(0, 1.1)
    st.pyplot(fig2)
    
    # Data warning for small samples
    if (ts_df['Review Count'] < 3).any():
        st.warning("⚠ Some time periods have very few reviews (<3). Consider using Quarterly view for more stable trends.")

with tab3:
    # Word clouds
    st.subheader("Word Clouds")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("*Positive Reviews*")
        positive_text = ' '.join(df[df['Sentiment'] == 'Positive']['Review'])
        wordcloud = WordCloud(width=400, height=300, background_color='white').generate(positive_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    
    with col2:
        st.markdown("*Negative Reviews*")
        negative_text = ' '.join(df[df['Sentiment'] == 'Negative']['Review'])
        wordcloud = WordCloud(width=400, height=300, background_color='white').generate(negative_text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    
    # Price vs Rating
    st.subheader("Price vs Rating")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='Price', y='Rating', hue='Sentiment', 
                   palette={'Positive':'green', 'Neutral':'blue', 'Negative':'red'}, ax=ax)
    st.pyplot(fig)

with tab4:
    # Prediction interface
    st.subheader("Live Sentiment Prediction")
    user_input = st.text_area("Enter a product review to analyze:", 
                            "This product exceeded my expectations!")
    
    if st.button("Predict Sentiment"):
        # Preprocess and predict
        text_processed = user_input.lower().replace('[^\w\s]', '')
        text_tfidf = tfidf.transform([text_processed])
        prediction_num = model.predict(text_tfidf)[0]
        sentiment = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}[prediction_num]
        
        # Display result with emoji
        emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😠"}[sentiment]
        if sentiment == 'Positive':
            st.success(f"Predicted Sentiment: {sentiment} {emoji}")
        elif sentiment == 'Neutral':
            st.info(f"Predicted Sentiment: {sentiment} {emoji}")
        else:
            st.error(f"Predicted Sentiment: {sentiment} {emoji}")

# Show raw data if selected
if show_raw_data:
    st.subheader("Raw Data")
    st.dataframe(filtered_df)
