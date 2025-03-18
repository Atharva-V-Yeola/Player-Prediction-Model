import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
import joblib
import streamlit as st
import shap

# Load dataset
df = pd.read_csv('football_matches.csv')

# Feature Engineering
features = ['home_team', 'away_team', 'home_score', 'away_score', 'possession', 'shots_on_target']
df = df[features]

# Encode categorical variables
label_encoder = LabelEncoder()
df['home_team'] = label_encoder.fit_transform(df['home_team'])
df['away_team'] = label_encoder.transform(df['away_team'])

# Define target variable
def get_match_result(row):
    if row['home_score'] > row['away_score']:
        return 'Win'
    elif row['home_score'] < row['away_score']:
        return 'Loss'
    else:
        return 'Draw'

df['match_result'] = df.apply(get_match_result, axis=1)
df['match_result'] = label_encoder.fit_transform(df['match_result'])

# Clustering teams based on playing styles
kmeans = KMeans(n_clusters=3, random_state=42)
df['team_cluster'] = kmeans.fit_predict(df[['home_team', 'away_team', 'possession', 'shots_on_target']])

# Split data
X = df.drop(columns=['match_result'])
y = df['match_result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter Tuning for Random Forest
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)
rf_model = grid_search.best_estimator_
rf_pred = rf_model.predict(X_test)
print("Optimized Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

# Feature Importance using SHAP
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# Train LSTM Model for Time-Series Prediction
X_lstm = X.values.reshape((X.shape[0], X.shape[1], 1))
X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm = train_test_split(X_lstm, y, test_size=0.2, random_state=42)

lstm_model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dense(3, activation='softmax')
])

lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=32, validation_data=(X_test_lstm, y_test_lstm))

# Save models
joblib.dump(rf_model, 'random_forest_model.pkl')
lstm_model.save('lstm_model.h5')

# Deploy using Streamlit
st.title("Football Match Outcome Prediction")
home_team = st.selectbox("Select Home Team", df['home_team'].unique())
away_team = st.selectbox("Select Away Team", df['away_team'].unique())
possession = st.slider("Possession %", 0, 100, 50)
shots_on_target = st.slider("Shots on Target", 0, 20, 5)

if st.button("Predict Outcome"):
    input_data = np.array([[label_encoder.transform([home_team])[0], label_encoder.transform([away_team])[0], possession, shots_on_target]])
    input_data = scaler.transform(input_data)
    prediction = rf_model.predict(input_data)
    result = label_encoder.inverse_transform(prediction)
    st.write("Predicted Match Result:", result[0])
