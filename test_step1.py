"""
Test script for Step 1: Data Preprocessing
Run this to test your data processing pipeline
"""

# Add the src directory to Python path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import TwitterDataProcessor

def test_data_processing():
    """Test the data processing pipeline"""
    print("🧪 Testing Data Processing Pipeline")
    print("=" * 50)
    
    # Initialize processor
    processor = TwitterDataProcessor()
    
    # Check if data exists
    data_file = "data/tweets-engagement-metrics.csv"
    
    if not os.path.exists(data_file):
        print("📝 Creating sample data for testing...")
        
        # Create sample data based on your example
        import pandas as pd
        
        sample_data = {
            'TweetID': ['tw-698155297102295041', 'tw-685159757209059329', 'tw-686907710311378944'],
            'Hour': [7, 11, 6],
            'Day': [12, 7, 12],
            'Weekday': ['Friday', 'Thursday', 'Tuesday'],
            'IsReshare': [True, False, False],
            'Reach': [339, 87, 87],
            'RetweetCount': [127, 0, 0],
            'Likes': [0, 0, 0],
            'Klout': [44, 22, 22],
            'Sentiment': [0, 0, 0],
            'Lang': ['en', 'en', 'en'],
            'text': [
                'RT @AdrianRusso82: Our Innovation Lab is officially open! #Tech #JavaScript #AWS',
                'Now Open AWS Asia Pacific (Seoul) Region via /r/sysadmin #AWS #Cloud',
                'A Beginners Guide to Scaling to 11 Million+ Users on Amazon AWS #AWS #Scaling'
            ]
        }
        
        df = pd.DataFrame(sample_data)
        os.makedirs('data', exist_ok=True)
        df.to_csv(data_file, index=False)
        print("✅ Sample data created!")
    
    # Run processing
    try:
        # Load and process
        processor.load_data(data_file)
        processor.inspect_data()
        processor.clean_sensitive_data()
        processor.feature_engineering()
        processor.get_processing_summary()
        
        # Show some results
        print("\n📊 SAMPLE PROCESSED DATA:")
        print("=" * 30)
        cols_to_show = ['text', 'hashtags', 'hashtag_count', 'word_count', 'virality_score']
        available_cols = [col for col in cols_to_show if col in processor.processed_df.columns]
        print(processor.processed_df[available_cols].head())
        
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_data_processing()
    if success:
        print("\n🎉 Ready for Step 2: Machine Learning Model Development!")
    else:
        print("\n🔧 Please fix the issues above before proceeding.")
