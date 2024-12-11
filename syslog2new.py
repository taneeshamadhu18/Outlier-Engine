import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the data
file_path = r"C:\Users\tanee\Downloads\syslog.csv"
syslog_data = pd.read_csv(file_path)

# Fill missing values
syslog_data.fillna('', inplace=True)

# Convert timestamps to datetime objects for feature extraction
syslog_data['recv_date'] = pd.to_datetime(syslog_data['recv_date'])
syslog_data['occur_date'] = pd.to_datetime(syslog_data['occur_date'])

# Create additional features from the timestamps
syslog_data['hour'] = syslog_data['recv_date'].dt.hour
syslog_data['day_of_week'] = syslog_data['recv_date'].dt.dayofweek
syslog_data['is_night'] = syslog_data['hour'].apply(lambda x: 1 if x < 6 else 0)  # 1 if night, 0 if day

# Label the severity (failure prediction: 1 for high severity, 0 for low severity)
syslog_data['label'] = syslog_data['severity'].apply(lambda x: 1 if x > 4 else 0)

# Use event and message text for the model
syslog_data['text'] = syslog_data['event'] + ' ' + syslog_data['message']

# IP Address feature extraction (you can also try more advanced encoding here)
syslog_data['ip_prefix'] = syslog_data['ip'].apply(lambda x: x.split('.')[0])  # Extract first octet as prefix

# Convert categorical features (like ip_prefix) using Label Encoding or One-Hot Encoding
label_encoder = LabelEncoder()
syslog_data['ip_encoded'] = label_encoder.fit_transform(syslog_data['ip_prefix'])

# Use TF-IDF to vectorize the text data (syslog messages)
vectorizer = TfidfVectorizer(max_features=1000)
X_text = vectorizer.fit_transform(syslog_data['text']).toarray()

# Combine all features: numeric features + text features
X = np.hstack((X_text, syslog_data[['severity', 'hour', 'day_of_week', 'is_night', 'ip_encoded']].values))

# Target variable (label for failure prediction)
y = syslog_data['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Print classification results
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Stratified Cross-validation (for better generalization)
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
cross_val_scores = []

for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X[train_index], X[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_fold, y_train_fold)

    y_pred_fold = model.predict(X_test_fold)
    score = classification_report(y_test_fold, y_pred_fold, output_dict=True, zero_division=1)['accuracy']
    cross_val_scores.append(score)

print("Stratified Cross-Validation Accuracy:", np.mean(cross_val_scores))

# Save the trained model and vectorizer for later use
with open('router_failure_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

with open('label_encoder.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoder, encoder_file)

print("Model, vectorizer, and encoder saved successfully.")

# Function to load the saved model and make predictions on new logs
def load_model_and_predict(new_data):
    # Load the saved model, vectorizer, and label encoder
    with open('router_failure_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)

    with open('label_encoder.pkl', 'rb') as encoder_file:
        loaded_encoder = pickle.load(encoder_file)

    # Process the new data
    new_data_df = pd.DataFrame(new_data, columns=['event', 'message', 'ip', 'severity', 'recv_date'])
    new_data_df['recv_date'] = pd.to_datetime(new_data_df['recv_date'])
    new_data_df['hour'] = new_data_df['recv_date'].dt.hour
    new_data_df['day_of_week'] = new_data_df['recv_date'].dt.dayofweek
    new_data_df['is_night'] = new_data_df['hour'].apply(lambda x: 1 if x < 6 else 0)
    new_data_df['ip_prefix'] = new_data_df['ip'].apply(lambda x: x.split('.')[0])
    new_data_df['ip_encoded'] = loaded_encoder.transform(new_data_df['ip_prefix'])

    new_data_df['text'] = new_data_df['event'] + ' ' + new_data_df['message']
    new_data_text = loaded_vectorizer.transform(new_data_df['text']).toarray()
    new_data_features = np.hstack((new_data_text, new_data_df[['severity', 'hour', 'day_of_week', 'is_night', 'ip_encoded']].values))

    # Get prediction probabilities for each class
    probabilities = loaded_model.predict_proba(new_data_features)

    # Print the feature values and predicted probabilities
    for i, log in new_data_df.iterrows():
        print(f"Log {i + 1}:")
        print(f"  Event: {log['event']}")
        print(f"  Message: {log['message']}")
        print(f"  IP Address: {log['ip']}")
        print(f"  Severity: {log['severity']}")
        print(f"  Hour: {log['hour']}")
        print(f"  Day of Week: {log['day_of_week']}")
        print(f"  Is Night: {log['is_night']}")
        print(f"  IP Encoded: {log['ip_encoded']}")
        print(f"  Predicted Probabilities (Failure vs. Non-Failure): {probabilities[i]}")
        print(f"  Predicted Class: {np.argmax(probabilities[i])}")
        print("-" * 50)

# Example new log data for prediction
new_logs = [
    {"event": "CONFIG_I", "message": "Configured from console by console", "ip": "192.168.100.5", "severity": 5, "recv_date": "1900-09-10 23:00:33"},
    {"event": "CPUHOG", "message": "Task is running for 2500ms, more than 2000ms", "ip": "192.168.100.5", "severity": 3, "recv_date": "1900-09-10 19:58:22"}
]

# Call the prediction function for the new logs
load_model_and_predict(new_logs)
