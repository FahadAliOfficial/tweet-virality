"""
Model Analysis and Extended Evaluation
Twitter Virality Prediction - Comprehensive Performance Analysis
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

def analyze_model_performance():
    """
    Comprehensive analysis of the trained model including feature importance,
    prediction accuracy, and precision-like metrics for regression.
    """
    print("🔍 Starting Comprehensive Model Analysis...")
    print("=" * 60)
    
    # --- Load Data and Model ---
    splits_dir = "data/splits"
    model_path = "models/xgb_virality_predictor.joblib"
    
    print("📂 Loading data and trained model...")
    try:
        X_train = pd.read_csv(os.path.join(splits_dir, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(splits_dir, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(splits_dir, "y_train.csv")).values.ravel()
        y_test = pd.read_csv(os.path.join(splits_dir, "y_test.csv")).values.ravel()
        
        model = joblib.load(model_path)
        print("✅ Data and model loaded successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure you have run training first.")
        return
    
    # --- Make Predictions ---
    print("\n🎯 Making predictions...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # --- Comprehensive Evaluation Metrics ---
    print("\n📊 COMPREHENSIVE PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Standard regression metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_mse = mean_squared_error(y_train, y_pred_train)
    train_r2 = r2_score(y_train, y_pred_train)
    train_rmse = np.sqrt(train_mse)
    
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_mse = mean_squared_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    test_rmse = np.sqrt(test_mse)
    
    print("📈 Standard Regression Metrics:")
    print(f"  Training Set:")
    print(f"    - R² Score:     {train_r2:.4f} ({train_r2*100:.2f}%)")
    print(f"    - MAE:          {train_mae:.4f}")
    print(f"    - RMSE:         {train_rmse:.4f}")
    print(f"  Testing Set:")
    print(f"    - R² Score:     {test_r2:.4f} ({test_r2*100:.2f}%)")
    print(f"    - MAE:          {test_mae:.4f}")
    print(f"    - RMSE:         {test_rmse:.4f}")
    
    # --- Precision-like Metrics for Regression ---
    print(f"\n🎯 Precision-like Metrics for Regression:")
    
    # Calculate prediction accuracy within different tolerance ranges
    tolerances = [0.1, 0.25, 0.5, 1.0]
    
    for tolerance in tolerances:
        # Percentage of predictions within tolerance
        within_tolerance = np.abs(y_test - y_pred_test) <= tolerance
        accuracy_pct = np.mean(within_tolerance) * 100
        print(f"    - Accuracy within ±{tolerance:.2f}: {accuracy_pct:.2f}%")
    
    # Mean Absolute Percentage Error (MAPE) - for relative accuracy
    # Note: We add 1 to avoid division by zero (since we're using log scale)
    mape = np.mean(np.abs((y_test - y_pred_test) / (y_test + 1))) * 100
    print(f"    - Mean Absolute Percentage Error: {mape:.2f}%")
    
    # --- Transform back to original scale for interpretation ---
    print(f"\n📊 Real-world Interpretation (Original Scale):")
    # Convert log predictions back to original virality scores
    y_test_original = np.expm1(y_test)  # inverse of log1p
    y_pred_original = np.expm1(y_pred_test)
    
    original_mae = mean_absolute_error(y_test_original, y_pred_original)
    original_r2 = r2_score(y_test_original, y_pred_original)
    
    print(f"    - R² on original scale: {original_r2:.4f} ({original_r2*100:.2f}%)")
    print(f"    - MAE on original scale: {original_mae:.2f} virality points")
    print(f"    - Average actual virality: {np.mean(y_test_original):.2f}")
    print(f"    - Average predicted virality: {np.mean(y_pred_original):.2f}")
    
    # --- Feature Importance Analysis ---
    print(f"\n🏆 TOP 10 MOST IMPORTANT FEATURES:")
    feature_importance = model.feature_importances_
    feature_names = X_train.columns
    
    # Create feature importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Display top 10 features
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"    {i:2d}. {row['feature']:<20} {row['importance']:.4f}")
    
    # --- Prediction Quality Analysis ---
    print(f"\n📊 Prediction Quality Analysis:")
    
    # Calculate prediction errors
    errors = y_test - y_pred_test
    
    print(f"    - Mean prediction error: {np.mean(errors):.4f}")
    print(f"    - Std of prediction errors: {np.std(errors):.4f}")
    print(f"    - 95% of predictions within: ±{np.percentile(np.abs(errors), 95):.4f}")
    
    # Overfitting check
    overfitting_gap = train_r2 - test_r2
    print(f"    - Overfitting gap (train R² - test R²): {overfitting_gap:.4f}")
    
    if overfitting_gap < 0.05:
        print("    ✅ Low overfitting - Good generalization!")
    elif overfitting_gap < 0.1:
        print("    ⚠️  Moderate overfitting - Acceptable")
    else:
        print("    ❌ High overfitting - Consider regularization")
    
    # --- Performance Categories ---
    print(f"\n🏅 OVERALL MODEL PERFORMANCE RATING:")
    
    if test_r2 >= 0.8:
        rating = "EXCELLENT"
        emoji = "🏆"
    elif test_r2 >= 0.7:
        rating = "VERY GOOD"
        emoji = "🥇"
    elif test_r2 >= 0.6:
        rating = "GOOD"
        emoji = "🥈"
    elif test_r2 >= 0.5:
        rating = "FAIR"
        emoji = "🥉"
    else:
        rating = "NEEDS IMPROVEMENT"
        emoji = "📈"
    
    print(f"    {emoji} {rating} ({test_r2*100:.1f}% accuracy)")
    
    # --- Business Impact Assessment ---
    print(f"\n💼 BUSINESS IMPACT ASSESSMENT:")
    print(f"    ✅ Model explains {test_r2*100:.1f}% of virality variance")
    print(f"    ✅ Average prediction error: {test_mae:.2f} log points")
    print(f"    ✅ 95% predictions within: ±{np.percentile(np.abs(errors), 95):.2f} log points")
    
    # Real-world accuracy interpretation
    within_50pct = np.mean(np.abs(errors) <= 0.5) * 100
    print(f"    ✅ {within_50pct:.1f}% of predictions within reasonable range (±0.5)")
    
    if within_50pct >= 70:
        print("    🎯 READY FOR PRODUCTION - High prediction reliability!")
    elif within_50pct >= 60:
        print("    ⚠️  GOOD FOR BETA - Reliable with some variance")
    else:
        print("    📊 NEEDS IMPROVEMENT - Consider more features or data")
    
    print(f"\n🎉 Analysis Complete! Your model is performing at {rating} level.")
    
    return {
        'test_r2': test_r2,
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'mape': mape,
        'feature_importance': importance_df,
        'within_tolerances': {tol: np.mean(np.abs(y_test - y_pred_test) <= tol) * 100 
                             for tol in tolerances}
    }

if __name__ == "__main__":
    results = analyze_model_performance()
