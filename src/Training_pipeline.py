import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

def train_and_evaluate_model():
    """
    Trains an XGBoost model on the training data, evaluates it on the testing data,
    and saves the trained model.
    """
    print("🚀 Starting Model Training & Evaluation...")
    print("==================================================")

    # --- Load Data Splits ---
    splits_dir = "data/splits"
    print(f"🔄 Loading data splits from '{splits_dir}'...")
    
    try:
        X_train = pd.read_csv(os.path.join(splits_dir, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(splits_dir, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(splits_dir, "y_train.csv")).values.ravel()
        y_test = pd.read_csv(os.path.join(splits_dir, "y_test.csv")).values.ravel()
    except FileNotFoundError as e:
        print(f"❌ Error: {e}. Make sure you have run the data splitting script first.")
        return

    print("✅ Data loaded successfully.")
    print(f"  - Training features shape: {X_train.shape}")
    print(f"  - Testing features shape:  {X_test.shape}")    # --- Model Training ---
    print("\n🧠 Training the XGBoost Regressor model...")
    
    # Initialize the XGBoost Regressor with some sensible defaults
    xgb_reg = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,           # Reduced number of boosting rounds
        learning_rate=0.1,          # Slightly higher learning rate
        max_depth=6,                # Maximum depth of a tree
        subsample=0.8,              # Subsample ratio of the training instance
        colsample_bytree=0.8,       # Subsample ratio of columns when constructing each tree
        random_state=42,            # for reproducibility
        n_jobs=-1                   # Use all available CPU cores
    )
    
    # Train the model
    print("  Training in progress...")
    xgb_reg.fit(X_train, y_train)
    
    print("✅ Model training complete.")

    # --- Model Evaluation ---
    print("\n🧪 Evaluating the model on the test set...")
    y_pred = xgb_reg.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("📊 Evaluation Metrics:")
    print(f"  - Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  - Mean Squared Error (MSE):  {mse:.4f}")
    print(f"  - R-squared (R²):            {r2:.4f}")

    # --- Save the Model ---
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "xgb_virality_predictor.joblib")
    
    print(f"\n💾 Saving the trained model to '{model_path}'...")
    joblib.dump(xgb_reg, model_path)
    print("✅ Model saved successfully.")

    print("\n🎉 Model training and evaluation finished!")
    print("✨ You now have a trained model ready for making predictions.")

if __name__ == "__main__":
    train_and_evaluate_model()