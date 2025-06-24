"""
Step 2: Data Splitting
Twitter Virality Prediction - Data Splitting Module
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_data(input_path="data/processed_twitter_data.csv", test_size=0.2, random_state=42):
    """
    Load the processed data, split it into features and target,
    and then into training and testing sets.
    """
    print("🚀 Starting Data Splitting...")
    print("=" * 50)
    
    # Check if the processed data file exists
    if not os.path.exists(input_path):
        print(f"❌ Error: Processed data file not found at '{input_path}'")
        print("Please run the data processing script first.")
        return

    # Load the processed dataset
    print(f"🔄 Loading processed data from '{input_path}'...")
    df = pd.read_csv(input_path)
    print("✅ Data loaded successfully.")
    print(f"📊 Dataset shape: {df.shape}")
    
    # --- Feature Selection ---
    # Define the target variable
    target = 'log_virality_score'
    
    # Define columns to drop to create the feature set (X)
    # We drop identifiers, raw text, non-encoded categoricals, and the target + its variations
    cols_to_drop = [
        'Unnamed: 0', 'UserID', 'TweetID', 'text', 'hashtags', 'mentions', 
        'urls', 'clean_text', 'Gender', 'LocationID', 'City', 'State', 
        'StateCode', 'Country', 'Weekday', 'Lang', 'time_category',
        'Reach', 'RetweetCount', 'Likes', 'virality_score',
        'log_reach', 'log_likes', 'log_retweetcount', 'log_virality_score'
    ]
    
    # Handle cases where some columns might be missing in the input file
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    missing_cols = [col for col in cols_to_drop if col not in df.columns]
    
    if missing_cols:
        print(f"⚠️  Note: Some expected columns not found: {missing_cols}")
    
    print("\n🎯 Defining features (X) and target (y)...")
    X = df.drop(columns=existing_cols_to_drop)
    y = df[target]
    
    print(f"  - Target variable (y): '{target}'")
    print(f"  - Number of features (X): {X.shape[1]}")
    print(f"  - Features include: {list(X.columns)}")
    
    # Check for missing values in features
    missing_in_X = X.isnull().sum().sum()
    missing_in_y = y.isnull().sum()
    
    if missing_in_X > 0:
        print(f"⚠️  Warning: {missing_in_X} missing values found in features")
    if missing_in_y > 0:
        print(f"⚠️  Warning: {missing_in_y} missing values found in target")
    
    if missing_in_X == 0 and missing_in_y == 0:
        print("✅ No missing values detected in features or target")

    # --- Data Splitting ---
    print("\n🔪 Splitting data into training and testing sets...")
    print(f"  - Test size: {test_size*100}%")
    print(f"  - Random state: {random_state}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )
    
    print("✅ Data split successfully!")
    print(f"  - Training set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"  - Testing set shape:  X_test={X_test.shape}, y_test={y_test.shape}")
    
    # Display some statistics about the splits
    print(f"\n📊 Target Variable Statistics:")
    print(f"  - Full dataset - Mean: {y.mean():.3f}, Std: {y.std():.3f}")
    print(f"  - Training set - Mean: {y_train.mean():.3f}, Std: {y_train.std():.3f}")
    print(f"  - Testing set  - Mean: {y_test.mean():.3f}, Std: {y_test.std():.3f}")
    
    # --- Save the Splits ---
    output_dir = "data/splits"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n💾 Saving data splits to '{output_dir}' directory...")
    
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    # Also save feature names for later use
    feature_names_path = os.path.join(output_dir, "feature_names.txt")
    with open(feature_names_path, 'w') as f:
        for feature in X.columns:
            f.write(f"{feature}\n")
    
    print("✅ All data splits saved successfully:")
    print(f"  - X_train.csv: {X_train.shape}")
    print(f"  - X_test.csv: {X_test.shape}")
    print(f"  - y_train.csv: {y_train.shape}")
    print(f"  - y_test.csv: {y_test.shape}")
    print(f"  - feature_names.txt: {len(X.columns)} features")
    
    return X_train, X_test, y_train, y_test

def load_splits(splits_dir="data/splits"):
    """
    Load previously saved data splits for model training
    """
    print("🔄 Loading saved data splits...")
    
    try:
        X_train = pd.read_csv(os.path.join(splits_dir, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(splits_dir, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(splits_dir, "y_train.csv")).squeeze()
        y_test = pd.read_csv(os.path.join(splits_dir, "y_test.csv")).squeeze()
        
        print("✅ Data splits loaded successfully!")
        print(f"  - Training set: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"  - Testing set: X_test={X_test.shape}, y_test={y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    except FileNotFoundError as e:
        print(f"❌ Error loading splits: {e}")
        print("Please run the data splitting first.")
        return None, None, None, None

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = split_data()
    
    print("\n🎉 Data splitting completed!")
    print("📊 Your data is now ready for the next step: model training and evaluation.")
