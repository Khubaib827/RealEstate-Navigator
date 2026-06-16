import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_housing_model():
    print("🚀 Initializing Real Estate Model Training...")
    
    # Path configuration
    dataset_path = "Multan_Housing_Dataset_2000.csv"
    model_output = "house_price_model.pkl"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Missing essential dataset file: {dataset_path}")
        
    # Read dataset
    df = pd.read_csv(dataset_path)
    
    # Feature engineering split
    X = df.drop("House_Price_PKR", axis=1)
    y = df["House_Price_PKR"]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train robust Random Forest architecture
    print("⚙️ Training Random Forest Regressor Ensemble...")
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Validation evaluations
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n" + "="*30 + "\nMODEL EVALUATION STATISTICS\n" + "="*30)
    print(f"Mean Absolute Error (MAE): PKR {mae:,.2f}")
    print(f"R² Fit Score: {r2:.4f} ({round(r2*100, 2)}% variance explained)")
    print("="*30)
    
    # Serialize model assets
    joblib.dump(model, model_output)
    print(f"💾 Model artifact saved successfully as: '{model_output}'\n")

if __name__ == "__main__":
    train_housing_model()