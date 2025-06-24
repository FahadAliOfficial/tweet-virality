"""
Step 1: Data Preprocessing and Feature Engineering
Twitter Virality Prediction - Data Processing Module
"""

import pandas as pd
import numpy as np
import re
import warnings
from textblob import TextBlob
from datetime import datetime
import os

warnings.filterwarnings('ignore')

class TwitterDataProcessor:
    """
    A comprehensive data processor for Twitter virality prediction
    """
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        self.hashtags_list = []
        self.mentions_list = []
    
    def load_data(self, file_path=None):
        """Load the dataset and perform initial inspection"""
        if file_path:
            self.data_path = file_path
            
        print("🔄 Loading dataset...")
        
        # Try different separators and encodings
        try:
            # First try comma separated
            self.df = pd.read_csv(self.data_path, sep=',', encoding='utf-8')
            
            # Check if we have the expected columns
            expected_cols = ['TweetID', 'Hour', 'Day', 'Weekday', 'text']
            if not any(col in self.df.columns for col in expected_cols):
                # If not, the data might be in a single column, try to split it
                if len(self.df.columns) == 1:
                    print("🔧 Detected single-column format, attempting to parse...")
                    # Get the column name
                    col_name = self.df.columns[0]
                    # Split the data properly
                    self.df = pd.read_csv(self.data_path, sep=',', encoding='utf-8', header=0)
                    
        except:
            try:
                self.df = pd.read_csv(self.data_path, sep='\t', encoding='utf-8')
            except:
                try:
                    self.df = pd.read_csv(self.data_path, sep='\t', encoding='latin-1')
                except Exception as e:
                    print(f"❌ Error loading data: {e}")
                    return None
        
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Shape: {self.df.shape}")
        print(f"📋 Columns: {list(self.df.columns)}")
        
        return self.df
    
    def inspect_data(self):
        """Perform initial data inspection"""
        if self.df is None:
            print("❌ No data loaded. Please load data first.")
            return
            
        print("\n🔍 DATA INSPECTION REPORT")
        print("=" * 50)
        
        # Basic info
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Missing values
        print(f"\n📊 Missing Values:")
        missing = self.df.isnull().sum()
        for col, count in missing.items():
            if count > 0:
                print(f"  {col}: {count} ({count/len(self.df)*100:.1f}%)")
        
        # Data types
        print(f"\n📝 Data Types:")
        print(self.df.dtypes)
        
        # Sample data
        print(f"\n📋 Sample Data (First 3 rows):")
        print(self.df.head(3))
        
        # Target variables stats
        if 'Reach' in self.df.columns:
            print(f"\n🎯 Target Variables Statistics:")
            targets = ['Reach', 'RetweetCount', 'Likes']
            for target in targets:
                if target in self.df.columns:                    print(f"  {target}: Mean={self.df[target].mean():.1f}, Max={self.df[target].max()}")
    
    def clean_sensitive_data(self):
        """Remove or anonymize sensitive information"""
        print("\n🔒 Cleaning sensitive data...")
        
        if self.df is None:
            print("❌ No data to clean")
            return
        
        # Check for specific AWS key patterns only (more precise)
        sensitive_patterns = [
            r'AKIA[0-9A-Z]{16}',  # AWS Access Key ID - very specific pattern
            r'aws_secret_access_key',  # Explicit AWS secret key mentions
            r'AWS_SECRET_ACCESS_KEY',  # Case variations
        ]
        
        # Removed overly broad patterns that were catching too much data:
        # r'[0-9a-zA-Z/+]{40}' - too general, catches URLs, hashes, etc.
        # r'[A-Za-z0-9+/]{40}=*' - too general, catches base64 content
        
        removed_rows = 0
        for pattern in sensitive_patterns:
            if 'text' in self.df.columns:
                mask = self.df['text'].str.contains(pattern, regex=True, na=False)
                if mask.any():
                    print(f"⚠️  Found {mask.sum()} rows with potential sensitive data (pattern: {pattern[:10]}...)")
                    self.df = self.df[~mask]
                    removed_rows += mask.sum()
        
        if removed_rows > 0:
            print(f"🗑️  Removed {removed_rows} rows with sensitive data")
        else:
            print("✅ No sensitive data patterns found")
        
        # Reset index after removing rows
        self.df.reset_index(drop=True, inplace=True)
        
        return self.df
    
    def extract_hashtags(self, text):
        """Extract hashtags from text"""
        if pd.isna(text):
            return []
        hashtags = re.findall(r'#\w+', str(text))
        return [tag.lower() for tag in hashtags]
    
    def extract_mentions(self, text):
        """Extract mentions from text"""
        if pd.isna(text):
            return []
        mentions = re.findall(r'@\w+', str(text))
        return [mention.lower() for mention in mentions]
    
    def extract_urls(self, text):
        """Extract URLs from text"""
        if pd.isna(text):
            return []
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', str(text))
        return urls
    
    def clean_text(self, text):
        """Clean text by removing URLs, mentions, and special characters"""
        if pd.isna(text):
            return ""
        
        # Convert to string
        text = str(text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove mentions and hashtags for clean text
        text = re.sub(r'[@#]\w+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        return text.strip()
    
    def feature_engineering(self):
        """Create new features from existing data"""
        print("\n🔧 Feature Engineering...")
        
        if self.df is None:
            print("❌ No data loaded")
            return
        
        # Create a copy for processing
        self.processed_df = self.df.copy()
        
        # Text-based features
        if 'text' in self.processed_df.columns:
            print("  📝 Processing text features...")
            
            # Extract hashtags, mentions, URLs
            self.processed_df['hashtags'] = self.processed_df['text'].apply(self.extract_hashtags)
            self.processed_df['mentions'] = self.processed_df['text'].apply(self.extract_mentions)
            self.processed_df['urls'] = self.processed_df['text'].apply(self.extract_urls)
            
            # Count features
            self.processed_df['hashtag_count'] = self.processed_df['hashtags'].apply(len)
            self.processed_df['mention_count'] = self.processed_df['mentions'].apply(len)
            self.processed_df['url_count'] = self.processed_df['urls'].apply(len)
            
            # Clean text
            self.processed_df['clean_text'] = self.processed_df['text'].apply(self.clean_text)
            
            # Text length features
            self.processed_df['text_length'] = self.processed_df['text'].str.len()
            self.processed_df['clean_text_length'] = self.processed_df['clean_text'].str.len()
            self.processed_df['word_count'] = self.processed_df['clean_text'].apply(lambda x: len(str(x).split()))
          # Time-based features
        if 'Weekday' in self.processed_df.columns:
            print("  📅 Processing time features...")
            
            # Create IsWeekend feature
            weekend_days = ['Saturday', 'Sunday']
            self.processed_df['IsWeekend'] = self.processed_df['Weekday'].isin(weekend_days)
            
            # Create time categories
            if 'Hour' in self.processed_df.columns:
                def categorize_hour(hour):
                    if 6 <= hour < 12:
                        return 'Morning'
                    elif 12 <= hour < 18:
                        return 'Afternoon'
                    elif 18 <= hour < 24:
                        return 'Evening'
                    else:
                        return 'Night'
                
                self.processed_df['time_category'] = self.processed_df['Hour'].apply(categorize_hour)
        
        # Handle missing location data
        print("  📍 Processing location features...")
        location_cols = ['City', 'State', 'StateCode', 'Country']
        for col in location_cols:
            if col in self.processed_df.columns:
                self.processed_df[col] = self.processed_df[col].fillna('Unknown')
        
        # Create location features
        if 'Country' in self.processed_df.columns:
            self.processed_df['is_US'] = (self.processed_df['Country'] == 'US').astype(int)
        
        # User features
        if 'Gender' in self.processed_df.columns:
            # Handle missing gender
            self.processed_df['Gender'] = self.processed_df['Gender'].fillna('Unknown')
            # Create binary features for gender
            self.processed_df['is_male'] = (self.processed_df['Gender'] == 'Male').astype(int)
            self.processed_df['is_female'] = (self.processed_df['Gender'] == 'Female').astype(int)
        
        # Engagement ratios
        print("  📊 Creating engagement features...")
        
        # Like-to-reach ratio
        if 'Likes' in self.processed_df.columns and 'Reach' in self.processed_df.columns:
            self.processed_df['like_rate'] = self.processed_df['Likes'] / (self.processed_df['Reach'] + 1)
        
        # Retweet-to-reach ratio
        if 'RetweetCount' in self.processed_df.columns and 'Reach' in self.processed_df.columns:
            self.processed_df['retweet_rate'] = self.processed_df['RetweetCount'] / (self.processed_df['Reach'] + 1)
          # Create virality score (combined metric)
        if all(col in self.processed_df.columns for col in ['Reach', 'Likes', 'RetweetCount']):
            self.processed_df['virality_score'] = (
                self.processed_df['Reach'] * 0.1 + 
                self.processed_df['Likes'] * 0.3 + 
                self.processed_df['RetweetCount'] * 0.6
            )
        
        # Log transform target variables for better ML performance
        print("  📊 Creating log-transformed targets...")
        target_cols = ['Reach', 'Likes', 'RetweetCount']
        for col in target_cols:
            if col in self.processed_df.columns:
                # Use log1p (log(1+x)) to handle zero values gracefully
                self.processed_df[f'log_{col.lower()}'] = np.log1p(self.processed_df[col])
        
        # Log transform virality score if it exists
        if 'virality_score' in self.processed_df.columns:
            self.processed_df['log_virality_score'] = np.log1p(self.processed_df['virality_score'])
        
        # Collect all hashtags for later use
        all_hashtags = []
        for hashtag_list in self.processed_df['hashtags']:
            all_hashtags.extend(hashtag_list)
        self.hashtags_list = list(set(all_hashtags))
        
        print(f"✅ Feature engineering completed!")
        print(f"  📊 Created {len(self.processed_df.columns) - len(self.df.columns)} new features")
        print(f"  🏷️  Found {len(self.hashtags_list)} unique hashtags")
        
        return self.processed_df
    
    def get_processing_summary(self):
        """Get a summary of the processing results"""
        if self.processed_df is None:
            print("❌ No processed data available")
            return
        
        print("\n📋 PROCESSING SUMMARY")
        print("=" * 50)
        
        print(f"Original dataset: {self.df.shape}")
        print(f"Processed dataset: {self.processed_df.shape}")
        
        print(f"\n🏷️  Hashtag Statistics:")
        print(f"  Total unique hashtags: {len(self.hashtags_list)}")
        print(f"  Average hashtags per tweet: {self.processed_df['hashtag_count'].mean():.2f}")
        print(f"  Max hashtags in a tweet: {self.processed_df['hashtag_count'].max()}")
        
        print(f"\n📝 Text Statistics:")
        print(f"  Average text length: {self.processed_df['text_length'].mean():.1f} characters")
        print(f"  Average word count: {self.processed_df['word_count'].mean():.1f} words")
        
        print(f"\n🎯 Target Variable Statistics:")
        targets = ['Reach', 'Likes', 'RetweetCount', 'virality_score']
        for target in targets:
            if target in self.processed_df.columns:
                mean_val = self.processed_df[target].mean()
                max_val = self.processed_df[target].max()
                print(f"  {target}: Mean={mean_val:.1f}, Max={max_val:.1f}")
        
        # Show most common hashtags
        if len(self.hashtags_list) > 0:
            from collections import Counter
            hashtag_counter = Counter()
            for hashtag_list in self.processed_df['hashtags']:
                hashtag_counter.update(hashtag_list)
            
            print(f"\n🔥 Top 10 Most Common Hashtags:")
            for hashtag, count in hashtag_counter.most_common(10):
                print(f"  {hashtag}: {count} times")
    
    def save_processed_data(self, output_path=None):
        """Save the processed dataset"""
        if self.processed_df is None:
            print("❌ No processed data to save")
            return
        
        if output_path is None:
            output_path = "data/processed_twitter_data.csv"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the processed data
        self.processed_df.to_csv(output_path, index=False)
        
        # Save hashtags list separately
        hashtags_path = output_path.replace('.csv', '_hashtags.txt')
        with open(hashtags_path, 'w', encoding='utf-8') as f:
            for hashtag in sorted(self.hashtags_list):
                f.write(f"{hashtag}\n")
        
        print(f"✅ Processed data saved to: {output_path}")
        print(f"✅ Hashtags list saved to: {hashtags_path}")


def main():
    """Main function to run the data processing"""
    print("🚀 Starting Twitter Data Preprocessing...")
    print("=" * 60)
    
    # Initialize processor
    processor = TwitterDataProcessor()
    
    # Check if data file exists
    data_file = "data/tweets-engagement-metrics.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("📝 Please place your dataset in the data/ folder")
        return
    
    # Process the data
    try:
        # Load data
        processor.load_data(data_file)
        
        # Inspect data
        processor.inspect_data()
        
        # Clean sensitive data
        processor.clean_sensitive_data()
        
        # Feature engineering
        processor.feature_engineering()
        
        # Show summary
        processor.get_processing_summary()
        
        # Save processed data
        processor.save_processed_data()
        
        print("\n🎉 Data preprocessing completed successfully!")
        print("📊 Your data is now ready for machine learning!")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        print("💡 Please check your dataset format and try again")


if __name__ == "__main__":
    main()
