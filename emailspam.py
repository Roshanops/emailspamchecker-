import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Function to load and preprocess dataset
def load_and_preprocess_data(filepath):
    # Load the dataset
    df = pd.read_csv(filepath)

    # Ensure the dataset contains the required columns 'text' and 'label'
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns!")
    
    # Convert labels to binary (spam = 1, ham = 0)
    df['label'] = df['label'].map({'spam': 1, 'ham': 0})

    # Split data into features (X) and labels (y)
    X = df['text']
    y = df['label']
    
    return X, y

# Function to vectorize the text data
def vectorize_data(X):
    vectorizer = CountVectorizer(stop_words='english')
    X_vectorized = vectorizer.fit_transform(X)
    return X_vectorized, vectorizer

# Function to train and evaluate the model
def train_and_evaluate(X_train, X_test, y_train, y_test):
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Evaluate the model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    return model

# Function to predict if a new email is spam or not
def predict_spam(model, vectorizer, new_email):
    new_email_vectorized = vectorizer.transform([new_email])
    prediction = model.predict(new_email_vectorized)

    if prediction[0] == 1:
        return "The email is SPAM."
    else:
        return "The email is NOT SPAM."

# Main function to tie everything together
def main():
    # Filepath to your dataset
    filepath = "your_dataset.csv"  # Replace with your file path
    
    # Load and preprocess the data
    X, y = load_and_preprocess_data(filepath)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorize the text data
    X_train_vectorized, vectorizer = vectorize_data(X_train)
    X_test_vectorized, _ = vectorize_data(X_test)
    
    # Train the model and evaluate
    model = train_and_evaluate(X_train_vectorized, X_test_vectorized, y_train, y_test)
    
    # Example usage: Predict a new email
    new_email = "Congratulations! You've won a $1000 gift card. Claim now."
    result = predict_spam(model, vectorizer, new_email)
    print(result)

if __name__ == "__main__":
    main()
