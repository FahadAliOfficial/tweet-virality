# 🐦 Twitter Virality Prediction App

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40.2-red.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.3-orange.svg)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn-green.svg)
<!--![License](https://img.shields.io/badge/License-MIT-lightgrey.svg) !-->

An advanced machine learning application that predicts Twitter post virality using a trained XGBoost model with 78.66% accuracy. Built with Streamlit for an interactive user experience, this app analyzes tweet content, user profiles, and timing to provide actionable insights for social media optimization.

## 🎯 Project Overview

This project implements an end-to-end machine learning pipeline that:
- **Predicts virality scores** for Twitter posts before publishing
- **Estimates reach, likes, and retweets** with high accuracy
- **Provides optimization suggestions** for better engagement
- **Analyzes content** including hashtags, mentions, sentiment, and timing
- **Offers real-time predictions** through an intuitive web interface

### 🏆 Key Achievements
- **📊 78.66% Prediction Accuracy** (R² = 0.7866)
- **📈 102K+ Training Tweets** with comprehensive feature engineering
- **🎯 17 Key Features** optimized for virality prediction
- **⚡ Real-time Predictions** with sub-second response times
- **🔍 Advanced Analytics** including feature importance and optimization tips

## 📁 Project Structure
```
twitter-virality/
├── 📊 data/                    # Dataset and processed files
│   ├── tweets-engagement-metrics.csv       # Original dataset (102K+ tweets)
│   ├── processed_twitter_data.csv          # Cleaned & feature-engineered data
│   ├── processed_twitter_data_hashtags.txt # Unique hashtags list (7,889 tags)
│   └── splits/                             # Train/test data splits
│       ├── X_train.csv     # Training features (81,626 × 17)
│       ├── X_test.csv      # Testing features (20,407 × 17)
│       ├── y_train.csv     # Training targets
│       ├── y_test.csv      # Testing targets
│       └── feature_names.txt # Feature list
├── 🧠 models/                  # Trained ML models
│   └── xgb_virality_predictor.joblib      # XGBoost model (78.66% accuracy)
├── 🔧 src/                     # Source code modules
│   ├── data_processor.py       # Data cleaning & feature engineering
│   ├── data_splitter.py        # Train/test splitting
│   ├── Training_pipeline.py    # Model training pipeline
│   └── model_analysis.py       # Comprehensive evaluation
├── 🌐 app.py                   # Main Streamlit application
├── 📋 requirements.txt         # Python dependencies (20 packages)
├── 🔬 test_step1.py           # Data processing test
└── 📚 README.md              # This comprehensive documentation
```

## 🚀 Quick Start

### 1. **Environment Setup**
```bash
# Navigate to project directory
cd D:\Projects\twitter-virality

# Activate virtual environment
D:/Projects/twitter-virality/.venv/Scripts/Activate.ps1

# Install all dependencies
pip install -r requirements.txt
```

### 2. **Run the Application**
```bash
# Launch the Streamlit app
streamlit run app.py

# Or alternatively
python -m streamlit run app.py
```

### 3. **Access the Application**
- Open your browser and navigate to `http://localhost:8501`
- Start predicting tweet virality instantly!

## 📊 Technical Specifications

### 🎯 Machine Learning Model
- **Algorithm**: XGBoost Regressor
- **Performance**: R² = 0.7866 (78.66% accuracy)
- **MAE**: 0.5958 (log scale)
- **RMSE**: 0.8193 (log scale)
- **Training Data**: 102,033 tweets
- **Features**: 17 optimized features
- **Target**: Log-transformed virality score

### 📈 Key Metrics & Terminology

#### Virality Score
- **Definition**: Composite engagement metric combining reach, likes, and retweets
- **Formula**: `virality_score = reach × 0.1 + likes × 0.3 + retweets × 0.6`
- **Log Transformation**: `log_virality_score = log(1 + virality_score)` for ML training
- **Scale**: 0-2000+ (higher = more viral potential)

#### Prediction Accuracy Levels
- **Low Virality**: Score < 100 (Limited reach)
- **Medium Virality**: Score 100-500 (Good engagement)
- **High Virality**: Score 500-1000 (Strong viral potential)
- **Viral**: Score 1000+ (High viral potential)

#### Feature Importance Rankings
1. **Like Rate History** (37.95%) - Historical like-to-reach ratio
2. **User Influence (Klout)** (27.50%) - Social media influence score (1-100)
3. **Retweet Rate History** (7.83%) - Historical retweet-to-reach ratio
4. **Content Type** (7.57%) - Hashtags, mentions, URL patterns
5. **Demographics** (5.89%) - Gender, location, timing factors

## 🔧 Development Pipeline

### ✅ Step 1: Data Preprocessing & Feature Engineering
**File**: `src/data_processor.py`

#### 📊 Dataset Processing
- **Original Dataset**: 102,062 tweets, 20 columns (81.75 MB)
- **Final Dataset**: 102,033 tweets, 42 columns (99.97% retention)
- **Data Cleaning**: Removed 29 rows with AWS credentials
- **Missing Data**: Handled location/gender nulls with intelligent defaults

#### 🔧 Feature Engineering (22 New Features)
```python
# Text Analysis Features
- hashtag_count: Number of hashtags (#tags)
- mention_count: Number of mentions (@users)
- url_count: Number of URLs
- text_length: Character count of original text
- clean_text_length: Character count after cleaning
- word_count: Word count in clean text
- sentiment: TextBlob sentiment polarity (-1 to 1)

# Time-based Features
- IsWeekend: Binary flag for weekend posts
- time_category: Morning/Afternoon/Evening/Night

# User Features
- is_US: Location flag for US users
- is_male, is_female: Gender encoding
- Klout: Social influence score (1-100)

# Engagement Features
- like_rate: likes / (reach + 1)
- retweet_rate: retweets / (reach + 1)
- virality_score: Composite engagement metric

# ML-Ready Targets (Log-transformed)
- log_reach, log_likes, log_retweetcount, log_virality_score
```

#### 📈 Dataset Statistics
```
Hashtags: 7,889 unique tags (avg 1.20/tweet, max 17/tweet)
Text: avg 195.2 chars, 12.3 words per tweet
Reach: mean 8,428 users (max 10.3M for viral content)
Engagement: mean 8.0 retweets, 0.1 likes per tweet
Virality Score: mean 847.7 (max 1.03M for super viral content)
```

#### 🔥 Top 10 Trending Hashtags
1. `#aws` - 27,148 occurrences
2. `#cloud` - 6,736 occurrences
3. `#job` - 4,390 occurrences
4. `#jobs` - 3,585 occurrences
5. `#bigdata` - 2,210 occurrences
6. `#devops` - 1,915 occurrences
7. `#cloudcomputing` - 1,832 occurrences
8. `#jobsearch` - 1,594 occurrences
9. `#azure` - 1,557 occurrences
10. `#amazon` - 1,556 occurrences

**Run Command**: `python src/data_processor.py`

### ✅ Step 2: Data Splitting
**File**: `src/data_splitter.py`

#### 📊 Split Configuration
- **Training Set**: 81,626 samples (80%)
- **Testing Set**: 20,407 samples (20%)
- **Target Variable**: `log_virality_score`
- **Random State**: 42 (reproducible results)
- **Stratification**: None (continuous target)

#### 🎯 Feature Selection (17 Features)
```python
Selected Features:
- Hour, Day, IsWeekend           # Timing optimization
- hashtag_count, mention_count, url_count  # Content metrics
- text_length, clean_text_length, word_count  # Text analysis
- Klout, Sentiment               # User influence & emotion
- is_male, is_female, is_US      # Demographics
- like_rate, retweet_rate        # Historical engagement
- IsReshare                      # Content type
```

**Run Command**: `python src/data_splitter.py`

### ✅ Step 3: Model Training
**File**: `src/Training_pipeline.py`

#### 🧠 XGBoost Configuration
```python
XGBRegressor Parameters:
- objective: 'reg:squarederror'
- n_estimators: 500
- learning_rate: 0.1
- max_depth: 6
- subsample: 0.8
- colsample_bytree: 0.8
- random_state: 42
- n_jobs: -1 (all CPU cores)
```

#### 📊 Training Results
- **Training Time**: ~2-3 minutes
- **R² Score**: 0.7866 (78.66% accuracy)
- **MAE**: 0.5958 (log scale)
- **RMSE**: 0.8193 (log scale)

**Run Command**: `python src/Training_pipeline.py`

### ✅ Step 4: Model Analysis
**File**: `src/model_analysis.py`

#### 🔍 Comprehensive Evaluation
```python
Performance Metrics:
- R² Score: 78.66% (VERY GOOD rating)
- MAE: 0.5958 (log scale)
- MAPE: 15.73% (Mean Absolute Percentage Error)
- Overfitting Gap: 0.0436 (excellent generalization)

Precision-like Metrics:
- Within ±0.10: 24.96% of predictions
- Within ±0.25: 46.89% of predictions
- Within ±0.50: 71.52% of predictions
- Within ±1.00: 89.70% of predictions
```

#### 🏆 Top 10 Feature Importance
1. **like_rate** (0.3795) - 37.95%
2. **Klout** (0.2750) - 27.50%
3. **retweet_rate** (0.0783) - 7.83%
4. **text_length** (0.0757) - 7.57%
5. **is_female** (0.0589) - 5.89%
6. **clean_text_length** (0.0387) - 3.87%
7. **word_count** (0.0340) - 3.40%
8. **Hour** (0.0269) - 2.69%
9. **hashtag_count** (0.0262) - 2.62%
10. **Sentiment** (0.0143) - 1.43%

**Run Command**: `python src/model_analysis.py`

### ✅ Step 5: Streamlit Application
**File**: `app.py`

#### 🌐 Web Application Features
- **Interactive UI**: Modern Twitter-inspired design
- **Real-time Predictions**: Instant virality scoring
- **Content Analysis**: Hashtag, mention, URL detection
- **Optimization Tips**: Actionable improvement suggestions
- **Feature Importance**: Visual explanation of prediction factors
- **Responsive Design**: Mobile-friendly interface

#### 🎨 UI Components
```python
Sidebar Inputs:
- Tweet text area (280 char limit)
- User profile (Klout score, gender)
- Timing controls (hour, day of week)
- Content type flags

Main Display:
- Virality score gauge (0-2000 scale)
- Estimated reach/likes/retweets
- Content analysis breakdown
- Optimization suggestions
- Feature importance visualization
```

**Run Command**: `streamlit run app.py`

## 📊 Application Features

### 🎯 Prediction Capabilities
- **Virality Score**: 0-2000+ scale with color-coded levels
- **Estimated Reach**: Predicted number of viewers
- **Estimated Likes**: Expected like count
- **Estimated Retweets**: Predicted retweet volume
- **Confidence Intervals**: Reliability indicators

### 🔍 Content Analysis
- **Text Metrics**: Character count, word count, readability
- **Hashtag Analysis**: Count, suggestions, trending tags
- **Mention Detection**: User mentions and networking potential
- **URL Analysis**: Link count and engagement impact
- **Sentiment Analysis**: TextBlob-powered emotion scoring

### 💡 Optimization Suggestions
- **Hashtag Optimization**: "Add hashtags for discoverability"
- **Content Length**: "Optimal length: 10-25 words"
- **Timing Recommendations**: "Post during business hours"
- **Weekend Adjustments**: "Consider weekday posting"
- **Link Guidelines**: "1-2 URLs optimal for engagement"
- **Mention Strategy**: "Tag relevant accounts for visibility"

### 📈 Analytics Dashboard
- **Performance Gauge**: Visual virality level indicator
- **Feature Impact**: Real-time importance scores
- **Historical Context**: Model training statistics
- **Trending Content**: Popular hashtags and topics

## 🛠️ Dependencies

### Core Libraries
```python
# Web Framework
streamlit==1.40.2          # Interactive web application

# Data Science
pandas==2.2.3              # Data manipulation
numpy==2.2.1               # Numerical computing
scikit-learn==1.6.0        # Machine learning toolkit

# Machine Learning
xgboost==2.1.3             # Gradient boosting
lightgbm==4.5.0            # Alternative boosting
joblib==1.4.2              # Model serialization

# Visualization
matplotlib==3.10.0         # Static plotting
seaborn==0.13.2            # Statistical visualization
plotly==5.24.1             # Interactive plotting

# Natural Language Processing
nltk==3.9.1                # NLP toolkit
textblob==0.18.0           # Sentiment analysis
wordcloud==1.9.4           # Word cloud generation

# Utilities
requests==2.32.3           # HTTP requests
python-dotenv==1.0.1       # Environment variables
beautifulsoup4==4.12.3     # Web scraping
regex==2024.11.6           # Regular expressions
openpyxl==3.1.5           # Excel file support

# AI Integration
google-generativeai==0.8.3 # Gemini API (future enhancement)
```

## 🎮 Usage Examples

### Example 1: Basic Tweet Analysis
```
Input: "Just launched my new #startup! Excited to change the world 🚀 #innovation #tech"

Output:
- Virality Score: 245 (Medium)
- Estimated Reach: 24 users
- Estimated Likes: 0 likes
- Estimated Retweets: 2 retweets
- Suggestions: "Add more hashtags", "Post during business hours"
```

### Example 2: High-Engagement Content
```
Input: "Breaking: Major #AI breakthrough! This could revolutionize #machinelearning #datascience"

Output:
- Virality Score: 1,247 (Viral)
- Estimated Reach: 124 users
- Estimated Likes: 0 likes
- Estimated Retweets: 12 retweets
- Suggestions: "Excellent hashtag usage", "Perfect timing"
```

### Example 3: Optimization Needed
```
Input: "Hello world"

Output:
- Virality Score: 45 (Low)
- Estimated Reach: 4 users
- Estimated Likes: 0 likes
- Estimated Retweets: 0 retweets
- Suggestions: "Add hashtags", "Include more content", "Mention relevant accounts"
```

## 🔬 Technical Deep Dive

### Model Architecture
The XGBoost model uses gradient boosting with the following pipeline:
1. **Input Processing**: 17 numerical features
2. **Tree Ensemble**: 500 decision trees with max depth 6
3. **Regularization**: Subsample (0.8) and column sampling (0.8)
4. **Output**: Log-transformed virality score
5. **Post-processing**: Exponential transformation back to original scale

### Feature Engineering Insights
- **Text Features**: Capture content richness and engagement signals
- **Temporal Features**: Exploit timing patterns for optimal reach
- **User Features**: Leverage historical engagement and demographics
- **Interaction Features**: Cross-feature relationships enhance predictions

### Prediction Pipeline
```python
1. Text Analysis → Extract hashtags, mentions, URLs
2. Feature Creation → 17 numerical features
3. Model Prediction → XGBoost regression
4. Score Transformation → exp(log_score) - 1
5. Metric Estimation → Reach, likes, retweets
6. Optimization → Actionable suggestions
```

## 🚀 Future Enhancements

### Planned Features
- [ ] **Advanced NLP**: BERT/GPT embeddings for content analysis
- [ ] **Real-time Data**: Live Twitter API integration
- [ ] **A/B Testing**: Post performance tracking and comparison
- [ ] **User Authentication**: Personal analytics dashboard
- [ ] **API Development**: REST API for third-party integrations
- [ ] **Mobile App**: React Native companion app
- [ ] **Multi-platform**: Support for Instagram, LinkedIn, TikTok

### Model Improvements
- [ ] **Ensemble Methods**: Combine XGBoost with Neural Networks
- [ ] **Feature Expansion**: Image analysis, video content, emoji sentiment
- [ ] **Temporal Modeling**: Time series analysis for trending topics
- [ ] **Personalization**: User-specific prediction models
- [ ] **Confidence Intervals**: Uncertainty quantification
<!--##
## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

 👨‍💻 Author

**Your Name** - AI/ML Engineer & Data Scientist
- 🐦 Twitter: [@your_handle](https://twitter.com/your_handle)
- 💼 LinkedIn: [Your Profile](https://linkedin.com/in/your-profile)
- 📧 Email: your.email@example.com
!-->
## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---
<!--
--## 📚 Citation

If you use this project in your research or work, please cite:

```bibtex
@software{twitter_virality_predictor,
  title={Twitter Virality Prediction App},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/twitter-virality}
}
```
!-->
---

**🎉 Ready to predict viral content? Launch the app and start optimizing your social media strategy!**

## Development Progress

### ✅ Step 1: Data Preprocessing and Feature Engineering (COMPLETED)

**🎯 Goal:** Clean the dataset and create features for machine learning models.

**📊 Dataset Overview:**
- **Original Size:** 102,062 tweets with 20 columns (81.75 MB)
- **Final Size:** 102,033 tweets with 42 columns (99.97% data retention)
- **Security Cleaning:** Removed 29 rows containing AWS credentials
- **Feature Engineering:** Created 22 new features from existing data

**📚 Dataset**

- [Data Set](https://www.kaggle.com/datasets/paakhim10/tweets-and-engagement-metrics)

**🔧 Features Created:**
- **Text Features:** Hashtags, mentions, URLs extraction
- **Count Features:** hashtag_count, mention_count, url_count, word_count
- **Text Processing:** clean_text, text_length, clean_text_length
- **Time Features:** IsWeekend, time_category (Morning/Afternoon/Evening/Night)
- **Location Features:** Missing data handling, is_US flag
- **User Features:** Gender encoding (is_male, is_female)
- **Engagement Features:** like_rate, retweet_rate, virality_score
- **ML-Ready Targets:** log_reach, log_likes, log_retweetcount, log_virality_score

**📈 Dataset Statistics:**
- **Hashtags:** 7,889 unique hashtags (avg 1.20 per tweet, max 17)
- **Text:** Average 195.2 characters, 12.3 words per tweet
- **Reach:** Mean 8,428 users (Max: 10.3M - highly viral content!)
- **Engagement:** Mean 8.0 retweets, 0.1 likes per tweet
- **Virality Score:** Mean 847.7 (Max: 1.03M for super viral content)

**🔥 Top Trending Hashtags:**
1. `#aws` - 27,148 times
2. `#cloud` - 6,736 times  
3. `#job` - 4,390 times
4. `#jobs` - 3,585 times
5. `#bigdata` - 2,210 times
6. `#devops` - 1,915 times
7. `#cloudcomputing` - 1,832 times
8. `#jobsearch` - 1,594 times
9. `#azure` - 1,557 times
10. `#amazon` - 1,556 times

**📁 Generated Files:**
- `data/processed_twitter_data.csv` - Clean dataset ready for ML training
- `data/processed_twitter_data_hashtags.txt` - Complete list of unique hashtags

**🚀 How to Run Data Processing:**
```bash
# Activate environment
D:/Projects/twitter-virality/.venv/Scripts/Activate.ps1

# Run data processor
python src/data_processor.py
```

### ✅ Step 2: Data Splitting (COMPLETED)

**🎯 Goal:** Split processed data into training and testing sets for machine learning.

**📊 Data Split Results:**
- **Source Dataset:** 102,033 tweets with 42 features
- **Target Variable:** `log_virality_score` (log-transformed virality metric)
- **Training Set:** 81,626 samples (80%)
- **Testing Set:** 20,407 samples (20%)
- **Features Selected:** 17 meaningful features for ML prediction

**🎯 Selected Features for ML Models:**
- **Time Features:** `Hour`, `Day`, `IsWeekend` - Optimal posting timing
- **Content Features:** `hashtag_count`, `mention_count`, `url_count`, `text_length`, `clean_text_length`, `word_count`
- **User Features:** `Klout`, `Sentiment`, `is_male`, `is_female`, `is_US` - Demographics & influence
- **Engagement Features:** `like_rate`, `retweet_rate`, `IsReshare` - Historical patterns

**📈 Target Variable Statistics:**
- **Mean Log Virality Score:** 4.138 (well-distributed)
- **Standard Deviation:** 1.773
- **Data Quality:** ✅ No missing values detected
- **Train/Test Balance:** Consistent distribution across splits

**📁 Generated Files:**
```
data/splits/
├── X_train.csv      # Training features (81,626 × 17)
├── X_test.csv       # Testing features (20,407 × 17)
├── y_train.csv      # Training targets (81,626)
├── y_test.csv       # Testing targets (20,407)
└── feature_names.txt # List of all 17 features
```

**🚀 How to Run Data Splitting:**
```bash
# Run data splitter (after data processing)
python src/data_splitter.py

# Or load existing splits in other scripts
from src.data_splitter import load_splits
X_train, X_test, y_train, y_test = load_splits()
```

### 🔄 Next Steps

**Step 3: Machine Learning Model Development**
- Build prediction models for virality scoring
- Compare algorithms: Random Forest, XGBoost, LightGBM
- Create prediction pipeline with confidence intervals
- Model validation and performance evaluation

**Step 4: Streamlit Application Development**
- Design user-friendly interface for post input
<!-- - Integrate Gemini API for topic generation !-->
- Build real-time prediction dashboard
- Add optimization suggestions and insights

## Development Steps

See the step-by-step development plan below.
