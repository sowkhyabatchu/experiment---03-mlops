import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Load dataset
data = pd.read_csv("data/dataset_yt_video.csv")

# Select features and target
features = ["likes", "dislikes", "comment_count", "category_id"]
target = "views"

# Remove rows with missing values
data = data[features + [target]].dropna()

X = data[features]
y = data[target]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Model Training Completed!")
print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Create model directory
os.makedirs("models", exist_ok=True)

# Save model
joblib.dump(model, "models/youtube_views_model.pkl")

print("Model saved to models/youtube_views_model.pkl")