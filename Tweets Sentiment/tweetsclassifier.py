import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('punkt')
    nltk.download('stopwords')


def download_sample_data():
    """
    Create a balanced sample dataset for testing
    """
    print("Creating a balanced sample dataset...")

    # Sample positive tweets
    positive_tweets = [
        "I love this product! It's amazing and works perfectly!",
        "Great experience with customer service, very helpful team!",
        "Awesome movie, would definitely recommend to friends!",
        "So happy with my purchase, exceeded expectations!",
        "Best decision ever, worth every penny!",
        "Excellent quality, highly satisfied!",
        "Fantastic service, will come back again!",
        "Wonderful experience, made my day!",
        "Perfect solution for my needs!",
        "Outstanding performance, very impressed!",
        "Loving the new features, great update!",
        "Beautiful design and easy to use!",
        "Highly recommend this to everyone!",
        "Exactly what I was looking for!",
        "Super fast delivery, thank you!",
        "Very intuitive interface, love it!",
        "Great value for money!",
        "Excellent customer support!",
        "Perfect fit, very comfortable!",
        "Amazing results, couldn't be happier!"
    ]

    # Sample negative tweets
    negative_tweets = [
        "Terrible experience, would not recommend to anyone!",
        "Worst customer service I've ever encountered!",
        "Product broke after 2 days, very disappointed!",
        "Complete waste of money, don't buy this!",
        "Horrible quality, fell apart immediately!",
        "Awful performance, constantly crashing!",
        "Very unhappy with this purchase!",
        "Poor design, difficult to use!",
        "Not worth the price at all!",
        "Extremely slow delivery!",
        "Bad experience overall!",
        "Defective product, had to return it!",
        "Frustrating to use, not user-friendly!",
        "Overpriced for what you get!",
        "Did not meet expectations at all!",
        "Annoying bugs and glitches!",
        "Unreliable service!",
        "Misleading advertisement!",
        "Low quality materials!",
        "Very dissatisfied with this!"
    ]

    # Create DataFrame
    data = []

    # Add positive tweets (target = 4)
    for tweet in positive_tweets:
        data.append({
            'target': 4,
            'text': tweet,
            'sentiment': 1
        })

    # Add negative tweets (target = 0)
    for tweet in negative_tweets:
        data.append({
            'target': 0,
            'text': tweet,
            'sentiment': 0
        })

    df = pd.DataFrame(data)

    print(f"Created sample dataset with {len(df)} tweets")
    print(f"Positive tweets: {len(positive_tweets)}")
    print(f"Negative tweets: {len(negative_tweets)}")

    return df


def clean_text(text):
    """
    Clean and preprocess text data
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove user mentions and hashtags
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)

    # Remove special characters and numbers, keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]

    return ' '.join(tokens)


def create_and_train_model(df):
    """
    Create features and train models
    """
    print("\nPreprocessing text data...")
    df['cleaned_text'] = df['text'].apply(clean_text)

    # Remove empty texts
    df = df[df['cleaned_text'].str.strip() != '']

    print(f"Data after cleaning: {len(df)} tweets")
    print(f"\nClass distribution:")
    print(f"Positive (1): {(df['sentiment'] == 1).sum()}")
    print(f"Negative (0): {(df['sentiment'] == 0).sum()}")

    # Create TF-IDF features
    print("\nCreating features...")
    tfidf = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8,
        stop_words='english'
    )

    X = tfidf.fit_transform(df['cleaned_text'])
    y = df['sentiment'].values

    print(f"Feature matrix shape: {X.shape}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Training classes - Positive: {(y_train == 1).sum()}, Negative: {(y_train == 0).sum()}")

    # Define classifiers
    classifiers = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='liblinear'
        ),
        'Naive Bayes': MultinomialNB(alpha=0.1),
        'Random Forest': RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            max_depth=15
        )
    }

    results = {}

    print("\n" + "=" * 50)
    print("TRAINING MODELS")
    print("=" * 50)

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")

        try:
            # Train
            clf.fit(X_train, y_train)

            # Predict
            y_pred = clf.predict(X_test)

            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            results[name] = {
                'classifier': clf,
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            }

            print(f"✓ Accuracy: {accuracy:.4f}")
            print(f"✓ Precision: {results[name]['precision']:.4f}")
            print(f"✓ Recall: {results[name]['recall']:.4f}")
            print(f"✓ F1-Score: {results[name]['f1_score']:.4f}")

        except Exception as e:
            print(f"✗ Error training {name}: {str(e)}")

    return results, tfidf, X_test, y_test


def test_predictions(model, tfidf, test_tweets=None):
    """
    Test the model on new tweets
    """
    if test_tweets is None:
        test_tweets = [
            "I absolutely love this! It's fantastic!",
            "This is the worst product I've ever bought!",
            "The service was okay, nothing special.",
            "Amazing quality, very happy with my purchase!",
            "Terrible experience, would not recommend.",
            "It works fine, I guess.",
            "Outstanding! Better than expected!",
            "Complete waste of money!"
        ]

    print("\n" + "=" * 50)
    print("PREDICTION TESTS")
    print("=" * 50)

    for tweet in test_tweets:
        cleaned = clean_text(tweet)
        features = tfidf.transform([cleaned])

        try:
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]

            sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"
            confidence = probabilities[prediction]

            print(f"\nTweet: {tweet}")
            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.2%}")
            print(f"Probabilities - Negative: {probabilities[0]:.2%}, Positive: {probabilities[1]:.2%}")

        except Exception as e:
            print(f"\nTweet: {tweet}")
            print(f"Error predicting: {str(e)}")


def main():
    """
    Main function to run the complete pipeline
    """
    print("=" * 60)
    print("TWEET SENTIMENT CLASSIFIER")
    print("=" * 60)

    # Option 1: Use sample data (for testing)
    print("\n1. Creating balanced sample dataset...")
    df = download_sample_data()

    # Option 2: If you have the actual dataset, use this instead:

    # Update this path to your actual dataset
    DATA_PATH = r'training.1600000.processed.noemoticon.csv'

    try:
        print("Loading dataset from file...")
        columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
        df = pd.read_csv(DATA_PATH, encoding='latin-1', names=columns, nrows=50000)

        # Convert target to binary
        df['sentiment'] = df['target'].apply(lambda x: 1 if x == 4 else 0)

        print(f"Loaded {len(df)} tweets")
        print(f"Positive: {(df['sentiment'] == 1).sum()}, Negative: {(df['sentiment'] == 0).sum()}")

        # Check if we have both classes
        if len(df['sentiment'].unique()) < 2:
            print("Warning: Dataset has only one class! Adding sample data...")
            sample_df = download_sample_data()
            df = pd.concat([df, sample_df], ignore_index=True)

    except FileNotFoundError:
        print("Dataset file not found. Using sample data instead...")
        df = download_sample_data()


    # Create and train model
    results, tfidf, X_test, y_test = create_and_train_model(df)

    if not results:
        print("\nNo models were successfully trained!")
        return None, None, None

    # Display results
    print("\n" + "=" * 50)
    print("FINAL RESULTS")
    print("=" * 50)

    results_df = pd.DataFrame(results).T
    print(results_df[['accuracy', 'precision', 'recall', 'f1_score']].round(4))

    # Get best model
    best_model_name = results_df['accuracy'].idxmax()
    best_model = results[best_model_name]['classifier']

    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"📊 Best Accuracy: {results[best_model_name]['accuracy']:.4f}")

    # Test predictions
    test_predictions(best_model, tfidf)

    return best_model, tfidf, results


def interactive_mode():
    """
    Interactive mode for real-time predictions
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Type 'quit' to exit")
    print("Enter tweets to analyze their sentiment:\n")

    # Train a model first
    df = download_sample_data()
    results, tfidf, _, _ = create_and_train_model(df)

    if not results:
        print("Failed to train model. Exiting interactive mode.")
        return

    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    model = results[best_model_name]['classifier']

    while True:
        try:
            tweet = input("\nEnter tweet: ").strip()

            if tweet.lower() == 'quit':
                print("Goodbye!")
                break

            if not tweet:
                continue

            # Clean and predict
            cleaned = clean_text(tweet)
            features = tfidf.transform([cleaned])

            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]

            sentiment = "😊 POSITIVE" if prediction == 1 else "😞 NEGATIVE"
            confidence = probabilities[prediction]

            print(f"Sentiment: {sentiment}")
            print(f"Confidence: {confidence:.2%}")

            if confidence < 0.7:
                print("Note: Low confidence prediction")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    # Run the main pipeline
    model, tfidf, results = main()

    # Optionally run interactive mode
    if model is not None:
        run_interactive = input("\nWould you like to enter interactive mode? (y/n): ").strip().lower()
        if run_interactive == 'y':
            interactive_mode()

    print("\n" + "=" * 60)
    print("PROGRAM COMPLETED SUCCESSFULLY")
    print("=" * 60)