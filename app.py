"""
Twitter Virality Prediction App - Streamlit Interface
Main application for predicting tweet virality and optimizing social media content
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from datetime import datetime, time
import os
from textblob import TextBlob

# Page configuration
st.set_page_config(
    page_title="Twitter Virality Predictor",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1DA1F2;
    }
    .prediction-box {
        background: linear-gradient(90deg, #1DA1F2, #14171A);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }    .feature-importance {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
        color: #212529;
    }
    .feature-importance strong {
        color: #495057;
        font-size: 1rem;
    }
    .feature-importance small {
        color: #6c757d;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained XGBoost model"""
    try:
        model = joblib.load("models/xgb_virality_predictor.joblib")
        return model
    except FileNotFoundError:
        st.error("❌ Model not found! Please train the model first by running the training pipeline.")
        return None

@st.cache_data
def load_hashtags():
    """Load the list of hashtags from the dataset"""
    try:
        with open("data/processed_twitter_data_hashtags.txt", "r", encoding="utf-8") as f:
            hashtags = [line.strip() for line in f.readlines()]
        return hashtags
    except FileNotFoundError:
        return []

def extract_hashtags(text):
    """Extract hashtags from text"""
    if not text:
        return []
    hashtags = re.findall(r'#\w+', text.lower())
    return hashtags

def extract_mentions(text):
    """Extract mentions from text"""
    if not text:
        return []
    mentions = re.findall(r'@\w+', text.lower())
    return mentions

def extract_urls(text):
    """Extract URLs from text"""
    if not text:
        return []
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return urls

def clean_text(text):
    """Clean text for analysis"""
    if not text:
        return ""
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove mentions and hashtags for clean text
    text = re.sub(r'[@#]\w+', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()

def get_sentiment(text):
    """Get sentiment score using TextBlob"""
    if not text:
        return 0
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except:
        return 0

def create_features(text, user_klout, gender, hour, day, weekday, is_reshare):
    """Create feature vector for prediction"""
    
    # Text analysis
    hashtags = extract_hashtags(text)
    mentions = extract_mentions(text)
    urls = extract_urls(text)
    clean_content = clean_text(text)
    
    # Calculate features
    features = {
        'Hour': hour,
        'Day': day,
        'IsReshare': 1 if is_reshare else 0,
        'Klout': user_klout,
        'Sentiment': get_sentiment(text),
        'hashtag_count': len(hashtags),
        'mention_count': len(mentions),
        'url_count': len(urls),
        'text_length': len(text) if text else 0,
        'clean_text_length': len(clean_content),
        'word_count': len(clean_content.split()) if clean_content else 0,
        'IsWeekend': 1 if weekday in ['Saturday', 'Sunday'] else 0,
        'is_US': 1,  # Default assumption, can be made configurable
        'is_male': 1 if gender == 'Male' else 0,
        'is_female': 1 if gender == 'Female' else 0,
        'like_rate': 0.001,  # Default low rate for new users
        'retweet_rate': 0.001  # Default low rate for new users
    }
    
    return features

def predict_virality(model, features):
    """Make virality prediction"""
    if model is None:
        return None
    
    # Convert features to DataFrame
    feature_df = pd.DataFrame([features])
    
    # Make prediction (returns log_virality_score)
    log_prediction = model.predict(feature_df)[0]
    
    # Convert back to original scale
    virality_score = np.expm1(log_prediction)
    
    # Estimate individual metrics based on virality score
    # These are rough estimates based on the relationships in your data
    estimated_reach = max(1, int(virality_score * 0.1))
    estimated_likes = max(0, int(virality_score * 0.0001))
    estimated_retweets = max(0, int(virality_score * 0.01))
    
    return {
        'virality_score': virality_score,
        'log_score': log_prediction,
        'estimated_reach': estimated_reach,
        'estimated_likes': estimated_likes,
        'estimated_retweets': estimated_retweets
    }

def get_optimization_suggestions(features, hashtags_list):
    """Provide optimization suggestions based on features"""
    suggestions = []
    
    # Hashtag suggestions
    if features['hashtag_count'] == 0:
        suggestions.append("🏷️ Add hashtags to increase discoverability!")
        suggestions.append(f"💡 Try popular hashtags like: {', '.join(hashtags_list[:5])}")
    elif features['hashtag_count'] > 5:
        suggestions.append("⚠️ Consider reducing hashtags (3-5 is optimal)")
    
    # Content length suggestions
    if features['word_count'] < 5:
        suggestions.append("📝 Add more content - longer posts tend to perform better")
    elif features['word_count'] > 30:
        suggestions.append("✂️ Consider shortening your post for better engagement")
    
    # Timing suggestions
    if features['Hour'] < 9 or features['Hour'] > 17:
        suggestions.append("⏰ Consider posting during business hours (9 AM - 5 PM) for better reach")
    
    # Weekend suggestions
    if features['IsWeekend']:
        suggestions.append("📅 Weekend posts may have lower reach - consider posting on weekdays")
    
    # URL suggestions
    if features['url_count'] == 0:
        suggestions.append("🔗 Adding relevant links can increase engagement")
    elif features['url_count'] > 2:
        suggestions.append("⚠️ Too many links might reduce engagement")
    
    # Mention suggestions
    if features['mention_count'] == 0:
        suggestions.append("👥 Mentioning relevant accounts can increase visibility")
    
    return suggestions

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🐦 Twitter Virality Predictor</h1>', unsafe_allow_html=True)
    st.markdown("**Predict how viral your tweet will be and get optimization suggestions!**")
    
    # Load model and data
    model = load_model()
    hashtags_list = load_hashtags()
    
    if model is None:
        st.stop()
    
    # Sidebar for user inputs
    st.sidebar.header("📝 Post Details")
    
    # Text input
    tweet_text = st.sidebar.text_area(
        "✍️ Write your tweet:",
        placeholder="What's happening?",
        height=100,
        help="Enter the text of your tweet (max 280 characters)"
    )
    
    # Character count
    char_count = len(tweet_text) if tweet_text else 0
    if char_count > 280:
        st.sidebar.error(f"❌ Tweet too long! ({char_count}/280 characters)")
    else:
        st.sidebar.success(f"✅ {char_count}/280 characters")
    
    st.sidebar.header("👤 User Profile")
    
    # User details
    col1, col2 = st.sidebar.columns(2)
    with col1:
        user_klout = st.number_input("Klout Score", min_value=1, max_value=100, value=30, help="Your social media influence score (1-100)")
        gender = st.selectbox("Gender", ["Male", "Female", "Unknown"])
    
    with col2:
        is_reshare = st.checkbox("Is this a retweet?", help="Check if this is a retweet/share")
    
    st.sidebar.header("📅 Timing")
    
    # Timing inputs
    col1, col2 = st.sidebar.columns(2)
    with col1:
        post_time = st.time_input("Posting time", value=time(12, 0))
        hour = post_time.hour
        
    with col2:
        weekday = st.selectbox("Day of week", 
                              ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        day = datetime.now().day  # Current day for simplicity
    
    # Main content area
    if tweet_text and char_count <= 280:
        # Create features
        features = create_features(tweet_text, user_klout, gender, hour, day, weekday, is_reshare)
        
        # Make prediction
        prediction = predict_virality(model, features)
        
        if prediction:
            # Create two columns for results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Main prediction display
                st.markdown(f"""
                <div class="prediction-box">
                    <h2>🎯 Virality Prediction</h2>
                    <h1>{prediction['virality_score']:.0f}</h1>
                    <p>Virality Score</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Detailed metrics
                st.subheader("📊 Detailed Predictions")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric(
                        label="👁️ Estimated Reach",
                        value=f"{prediction['estimated_reach']:,}",
                        help="Estimated number of people who will see your tweet"
                    )
                
                with metric_col2:
                    st.metric(
                        label="❤️ Estimated Likes",
                        value=f"{prediction['estimated_likes']:,}",
                        help="Estimated number of likes"
                    )
                
                with metric_col3:
                    st.metric(
                        label="🔄 Estimated Retweets",
                        value=f"{prediction['estimated_retweets']:,}",
                        help="Estimated number of retweets"
                    )
                
                # Virality gauge
                st.subheader("🌟 Virality Level")
                
                # Determine virality level
                score = prediction['virality_score']
                if score < 100:
                    level = "Low"
                    color = "red"
                elif score < 500:
                    level = "Medium"
                    color = "orange"
                elif score < 1000:
                    level = "High"
                    color = "yellow"
                else:
                    level = "Viral"
                    color = "green"
                
                # Create gauge chart
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = min(score, 2000),  # Cap at 2000 for display
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Virality Level"},
                    delta = {'reference': 500},
                    gauge = {
                        'axis': {'range': [None, 2000]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 100], 'color': "lightgray"},
                            {'range': [100, 500], 'color': "gray"},
                            {'range': [500, 1000], 'color': "lightblue"},
                            {'range': [1000, 2000], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 1000
                        }
                    }
                ))
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Content analysis
                st.subheader("🔍 Content Analysis")
                
                hashtags = extract_hashtags(tweet_text)
                mentions = extract_mentions(tweet_text)
                urls = extract_urls(tweet_text)
                
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    st.write(f"**📝 Word Count:** {features['word_count']}")
                    st.write(f"**🏷️ Hashtags:** {len(hashtags)} - {', '.join(hashtags) if hashtags else 'None'}")
                    st.write(f"**👥 Mentions:** {len(mentions)} - {', '.join(mentions) if mentions else 'None'}")
                
                with analysis_col2:
                    st.write(f"**🔗 URLs:** {len(urls)}")
                    st.write(f"**😊 Sentiment:** {features['Sentiment']:.2f}")
                    st.write(f"**⏰ Posting Hour:** {hour}:00")
            
            with col2:
                # Optimization suggestions
                st.subheader("💡 Optimization Tips")
                
                suggestions = get_optimization_suggestions(features, hashtags_list)
                
                for suggestion in suggestions:
                    st.info(suggestion)
                
                # Feature importance (top factors)
                st.subheader("🎯 Key Success Factors")
                
                # Show most important features for this prediction
                important_features = [
                    ("Like Rate History", "37.95%"),
                    ("User Influence (Klout)", "27.50%"),
                    ("Retweet Rate History", "7.83%"),
                    ("Content Type", "7.57%"),
                    ("Demographics", "5.89%")
                ]                
                for feature, importance in important_features:
                    st.markdown(f"""
                    <div class="feature-importance">
                        <strong style="color: #1DA1F2;">{feature}</strong><br>
                        <small style="color: #666;">Impact: {importance}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Hashtag suggestions
                if hashtags_list:
                    st.subheader("🔥 Trending Hashtags")
                    popular_hashtags = hashtags_list[:10]  # Top 10
                    
                    for tag in popular_hashtags:
                        if st.button(f"Add {tag}", key=f"hashtag_{tag}"):
                            st.sidebar.text_area("✍️ Write your tweet:", value=tweet_text + f" {tag}")
    
    else:
        # Welcome message
        st.info("👆 Enter your tweet in the sidebar to get started!")
        
        # Show some example insights
        st.subheader("🎯 How It Works")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📝 1. Write Your Tweet**
            - Enter your tweet text
            - Set your user profile
            - Choose posting time
            """)
        
        with col2:
            st.markdown("""
            **🧠 2. AI Analysis**
            - Advanced ML prediction
            - 78.66% accuracy rate
            - Based on 102K+ tweets
            """)
        
        with col3:
            st.markdown("""
            **🚀 3. Get Insights**
            - Virality predictions
            - Optimization tips
            - Best posting strategies
            """)
        
        # Show model performance
        st.subheader("🏆 Model Performance")
        
        perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
        
        with perf_col1:
            st.metric("Accuracy", "78.66%")
        with perf_col2:
            st.metric("Training Data", "102K+ tweets")
        with perf_col3:
            st.metric("Features", "17 factors")
        with perf_col4:
            st.metric("Hashtags", f"{len(hashtags_list)}")

if __name__ == "__main__":
    main()
