import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pickle
import numpy as np

file_path = r"C:\Users\tanee\Downloads\syslog_balanced.csv"
syslog_data = pd.read_csv(file_path)

syslog_data.fillna('', inplace=True)

X = syslog_data.drop(columns=['label'])  # Features (remove label column)
y = syslog_data['label']  # Labels


print("Original Label Distribution:")
print(syslog_data['label'].value_counts())


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

cross_val_scores = []
for train_index, test_index in skf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_fold, y_train_fold)

    y_pred_fold = model.predict(X_test_fold)
    score = classification_report(y_test_fold, y_pred_fold, output_dict=True, zero_division=1)['accuracy']
    cross_val_scores.append(score)

print("Stratified Cross-Validation Accuracy:", np.mean(cross_val_scores))


print("Label Distribution (After Balancing):")
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))


with open('router_failure_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)


def load_model_and_predict(new_data):
    with open('router_failure_model.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        loaded_vectorizer = pickle.load(vectorizer_file)

   
    new_data_vectorized = loaded_vectorizer.transform(new_data).toarray()

   
    new_data_vectorized_df = pd.DataFrame(new_data_vectorized, columns=loaded_vectorizer.get_feature_names_out())

    
    predictions = loaded_model.predict(new_data_vectorized_df)
    return predictions

new_logs = ["Router failed due to high memory usage", "Network is stable and no issues found"]
predictions = load_model_and_predict(new_logs)
print("Predictions for new logs:", predictions)
