# Twitter Virality Prediction App

## Project Structure
```
twitter-virality/
├── data/                    # Dataset and processed data files
├── models/                  # Trained ML models
├── src/                     # Source code modules
├── assets/                  # Static files, images, etc.
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables
└── README.md               # This file
```

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   - Add your Gemini API key to `.env` file
   - Place your dataset in the `data/` folder

3. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## Environment Usuage
    ```
     Navigate to project
    cd D:\Projects\twitter-virality

    # Activate environment
    D:/Projects/twitter-virality/.venv/Scripts/Activate.ps1

    # Now you can use shorter commands:
    python app.py
    streamlit run app.py
    pip install new_package
    ```
## Project Goals

Create a Twitter post virality prediction system that:
- Analyzes user input and predicts viral potential
- Suggests optimal posting times and hashtags
- Provides real-time statistical analysis
- Uses AI to generate relevant topics and tags

## Development Progress

### ✅ Step 1: Data Preprocessing and Feature Engineering (COMPLETED)

**🎯 Goal:** Clean the dataset and create features for machine learning models.

**📊 Dataset Overview:**
- **Original Size:** 102,062 tweets with 20 columns (81.75 MB)
- **Final Size:** 102,033 tweets with 42 columns (99.97% data retention)
- **Security Cleaning:** Removed 29 rows containing AWS credentials
- **Feature Engineering:** Created 22 new features from existing data

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
- Integrate Gemini API for topic generation
- Build real-time prediction dashboard
- Add optimization suggestions and insights

## Development Steps

See the step-by-step development plan below.
