"""
NLP Multi-Model Classification System
Ready to run in Visual Studio Code

Requirements:
pip install scikit-learn nltk pandas joblib numpy

Usage:
1. Install required packages
2. Run this script
3. Follow the interactive prompts
"""

import pandas as pd
import numpy as np
import json
import re
import joblib
import os
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Check and install required packages
def check_packages():
    """Check if required packages are installed"""
    required_packages = ['sklearn', 'nltk', 'pandas', 'joblib', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {missing_packages}")
        print("Please install with: pip install scikit-learn nltk pandas joblib numpy")
        return False
    return True

if not check_packages():
    exit(1)

# Import ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

# Simple NLP preprocessing (no NLTK required for basic version)
class SimpleTextPreprocessor:
    """Simple text preprocessing using only regex and built-in functions"""
    
    def __init__(self):
        # Common English stopwords
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 
            'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 
            'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 
            'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
            'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 
            'at', 'by', 'for', 'with', 'through', 'during', 'before', 'after', 'above', 
            'below', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
            'further', 'then', 'once'
        }
        
    def clean_text(self, text: str) -> str:
        """Clean text using regex"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text"""
        words = text.split()
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        return ' '.join(filtered_words)
    
    def preprocess(self, text: str) -> str:
        """Complete preprocessing pipeline"""
        cleaned = self.clean_text(text)
        return self.remove_stopwords(cleaned)

class NLPClassifier:
    """Main NLP Classification System"""
    
    def __init__(self):
        self.preprocessor = SimpleTextPreprocessor()
        self.models = {}
        self.vectorizers = {}
        self.pipelines = {}
        self.training_history = []
        self.is_trained = False
        
    def initialize_models(self):
        """Initialize models and vectorizers"""
        print("Initializing models...")
        
        # Initialize vectorizers
        self.vectorizers = {
            'tfidf': TfidfVectorizer(max_features=5000, ngram_range=(1, 2)),
            'count': CountVectorizer(max_features=5000, ngram_range=(1, 1))
        }
        
        # Initialize models
        self.models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'naive_bayes': MultinomialNB()
        }
        
        # Create pipelines
        self.pipelines = {}
        for vec_name, vectorizer in self.vectorizers.items():
            for model_name, model in self.models.items():
                pipeline_name = f"{vec_name}_{model_name}"
                self.pipelines[pipeline_name] = Pipeline([
                    ('vectorizer', vectorizer),
                    ('classifier', model)
                ])
        
        print(f"Initialized {len(self.pipelines)} model pipelines")
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Preprocess a list of texts"""
        return [self.preprocessor.preprocess(text) for text in texts]
    
    def train(self, texts: List[str], labels: List[str], test_size: float = 0.2):
        """Train all models"""
        print(f"\nTraining models with {len(texts)} samples...")
        
        # Preprocess texts
        processed_texts = self.preprocess_texts(texts)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=test_size, random_state=42
        )
        
        results = {}
        
        # Train each pipeline
        for pipeline_name, pipeline in self.pipelines.items():
            print(f"Training {pipeline_name}...")
            
            # Train
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3)
            
            results[pipeline_name] = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            print(f"  Accuracy: {accuracy:.4f}, CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Save training info
        training_info = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': len(texts),
            'n_train': len(X_train),
            'n_test': len(X_test),
            'results': results
        }
        self.training_history.append(training_info)
        self.is_trained = True
        
        return results
    
    def predict(self, texts: List[str], model_name: str = 'tfidf_logistic') -> Dict:
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Models must be trained first!")
        
        if model_name not in self.pipelines:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.pipelines.keys())}")
        
        # Preprocess texts
        processed_texts = self.preprocess_texts(texts)
        
        # Get pipeline
        pipeline = self.pipelines[model_name]
        
        # Make predictions
        predictions = pipeline.predict(processed_texts)
        
        # Get probabilities if available
        probabilities = None
        try:
            probabilities = pipeline.predict_proba(processed_texts)
        except AttributeError:
            pass
        
        return {
            'model': model_name,
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist() if probabilities is not None else None,
            'input_texts': texts
        }
    
    def retrain(self, new_texts: List[str], new_labels: List[str], model_name: str = 'tfidf_logistic'):
        """Retrain a specific model with new data"""
        if model_name not in self.pipelines:
            raise ValueError(f"Model {model_name} not found")
        
        print(f"Retraining {model_name} with {len(new_texts)} new samples...")
        
        # Preprocess new texts
        processed_texts = self.preprocess_texts(new_texts)
        
        # Retrain the pipeline
        pipeline = self.pipelines[model_name]
        pipeline.fit(processed_texts, new_labels)
        
        print(f"Retraining completed for {model_name}")
        
        # Log retraining
        retrain_info = {
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'n_new_samples': len(new_texts),
            'action': 'retrain'
        }
        self.training_history.append(retrain_info)
    
    def save_models(self, directory: str = 'saved_models'):
        """Save all models"""
        os.makedirs(directory, exist_ok=True)
        
        # Save pipelines
        for name, pipeline in self.pipelines.items():
            joblib.dump(pipeline, f"{directory}/{name}.joblib")
        
        # Save training history
        with open(f"{directory}/training_history.json", 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print(f"Models saved to {directory}/")
    
    def load_models(self, directory: str = 'saved_models'):
        """Load saved models"""
        if not os.path.exists(directory):
            print(f"Directory {directory} not found")
            return
        
        # Load pipelines
        for filename in os.listdir(directory):
            if filename.endswith('.joblib'):
                model_name = filename.replace('.joblib', '')
                self.pipelines[model_name] = joblib.load(f"{directory}/{filename}")
        
        # Load training history
        history_file = f"{directory}/training_history.json"
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                self.training_history = json.load(f)
        
        self.is_trained = len(self.pipelines) > 0
        print(f"Loaded {len(self.pipelines)} models from {directory}/")
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get model performance summary"""
        if not self.training_history:
            return pd.DataFrame()
        
        data = []
        for session in self.training_history:
            if 'results' in session:
                for model_name, metrics in session['results'].items():
                    data.append({
                        'timestamp': session['timestamp'],
                        'model': model_name,
                        'accuracy': metrics['accuracy'],
                        'cv_mean': metrics['cv_mean'],
                        'cv_std': metrics['cv_std']
                    })
        
        return pd.DataFrame(data)

def get_sample_data():
    """Get sample data for demonstration"""
    sample_texts = [
        "I absolutely love this product! It's amazing and works perfectly.",
        "This is the worst thing I've ever bought. Terrible quality and waste of money.",
        "The product is okay, nothing special but it does what it's supposed to do.",
        "Absolutely fantastic! I highly recommend this to everyone. Great value!",
        "Not worth the price. Poor build quality and doesn't work as expected.",
        "Great value for money. Very satisfied with my purchase and fast delivery.",
        "Average product. Could be better but it's acceptable for the price range.",
        "Outstanding quality and excellent customer service! Will buy again.",
        "Very disappointed with this purchase. Expected much better quality.",
        "Perfect! Exactly what I was looking for. Excellent build quality.",
        "Decent product but nothing extraordinary. Gets the job done.",
        "Excellent customer support and fast shipping. Product works great!",
        "Poor quality materials. Broke after just a few uses. Not recommended.",
        "Good product overall. Minor issues but generally satisfied with purchase.",
        "Terrible experience. Product arrived damaged and customer service unhelpful."
    ]
    
    sample_labels = [
        "positive", "negative", "neutral", "positive", "negative",
        "positive", "neutral", "positive", "negative", "positive",
        "neutral", "positive", "negative", "neutral", "negative"
    ]
    
    return sample_texts, sample_labels

def interactive_demo():
    """Interactive demonstration of the NLP system"""
    print("=" * 60)
    print("NLP Multi-Model Classification System")
    print("=" * 60)
    
    # Initialize system
    nlp_system = NLPClassifier()
    nlp_system.initialize_models()
    
    # Get sample data
    texts, labels = get_sample_data()
    
    print(f"\nUsing sample dataset with {len(texts)} reviews")
    print("Sample texts:")
    for i, text in enumerate(texts[:3]):
        print(f"  {i+1}. {text} -> {labels[i]}")
    print("  ...")
    
    # Train models
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    results = nlp_system.train(texts, labels)
    
    # Show performance summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    
    performance_df = nlp_system.get_performance_summary()
    if not performance_df.empty:
        print(performance_df.to_string(index=False))
    
    # Interactive prediction loop
    print("\n" + "="*50)
    print("INTERACTIVE PREDICTION")
    print("="*50)
    print("Enter text to classify (or 'quit' to exit)")
    
    while True:
        user_input = input("\nEnter text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        try:
            # Make prediction
            result = nlp_system.predict([user_input])
            prediction = result['predictions'][0]
            
            print(f"Prediction: {prediction}")
            
            # Show probabilities if available
            if result['probabilities']:
                probs = result['probabilities'][0]
                classes = ['negative', 'neutral', 'positive']  # Assuming these classes
                if len(probs) == len(classes):
                    print("Confidence scores:")
                    for class_name, prob in zip(classes, probs):
                        print(f"  {class_name}: {prob:.3f}")
        
        except Exception as e:
            print(f"Error: {e}")
    
    # Save models
    print("\n" + "="*50)
    print("SAVING MODELS")
    print("="*50)
    
    nlp_system.save_models()
    
    # Demonstrate retraining
    print("\n" + "="*50)
    print("RETRAINING DEMONSTRATION")
    print("="*50)
    
    new_texts = [
        "This product exceeded my expectations! Absolutely brilliant!",
        "Complete garbage. Don't waste your money on this junk."
    ]
    new_labels = ["positive", "negative"]
    
    print("Adding new training data:")
    for text, label in zip(new_texts, new_labels):
        print(f"  '{text}' -> {label}")
    
    nlp_system.retrain(new_texts, new_labels)
    
    print("\nRetraining completed!")
    print("\nDemo finished. Models saved to 'saved_models/' directory.")

def main():
    """Main function to run the NLP system"""
    try:
        interactive_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Make sure you have installed required packages:")
        print("pip install scikit-learn nltk pandas joblib numpy")

if __name__ == "__main__":
    main()
