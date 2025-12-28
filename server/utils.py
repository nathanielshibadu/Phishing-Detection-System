import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import tldextract
import re
import pickle
from tensorflow.keras.models import load_model

class Preprocessor:
    def __init__(self, max_tokens=5000, max_seq_len=100):
        """
        Optimized preprocessor with limited vocabulary and sequence length
        to prevent memory issues in Colab.
        """
        # Keep only top 5000 tokens and truncate/pad URLs to 100 tokens
        self.tokenizer = Tokenizer(num_words=max_tokens, oov_token="<OOV>")
        self.scaler = StandardScaler()
        self.sequences_len = max_seq_len

    def fit(self, df):
        urls = df['url']

        # Tokenizing the URLs
        print("Tokenizing the URLs...")
        self.tokenizer.fit_on_texts(urls)
        sequences = self.tokenizer.texts_to_sequences(urls)

        # Pad or truncate sequences to the fixed max length
        sequences = pad_sequences(sequences, maxlen=self.sequences_len, padding='post', truncating='post')

        # Extract additional lexical features
        print("Extracting features...")
        features = self.extract_features(df)

        print("Concatenating all features...")
        all_data = np.concatenate([sequences, features], axis=1)

        # Fit the scaler
        self.scaler.fit(all_data)
        return

    def transform(self, df):
        urls = df['url']

        # Tokenizing and padding URLs
        print("Tokenizing the URLs...")
        sequences = self.tokenizer.texts_to_sequences(urls)
        sequences = pad_sequences(sequences, maxlen=self.sequences_len, padding='post', truncating='post')

        # Extract lexical features
        print("Extracting features...")
        features = self.extract_features(df)

        print("Concatenating all features...")
        all_data = np.concatenate([sequences, features], axis=1)

        # Scale and return
        return self.scaler.transform(all_data)

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def extract_features(self, df):
        # Compute basic lexical URL-based features
        df['url_length'] = df['url'].apply(len)
        df['num_digits'] = df['url'].apply(lambda x: sum(c.isdigit() for c in x))
        df['num_letters'] = df['url'].apply(lambda x: sum(c.isalpha() for c in x))
        df['num_special_chars'] = df['url'].apply(lambda x: len(re.findall('[^a-zA-Z0-9]', x)))
        df['num_hyphens'] = df['url'].apply(lambda x: x.count('-'))
        df['num_subdomains'] = df['url'].apply(lambda x: x.count('.'))
        df['has_https'] = df['url'].apply(lambda x: 1 if x.startswith('https://') else 0)
        df['tld_length'] = df['url'].apply(lambda x: len(tldextract.extract(x).suffix) if '.' in x else 0)
        df['has_ip'] = df['url'].apply(lambda x: 1 if re.match(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", x) else 0)

        # Return only numeric feature columns as numpy array
        return np.array(df.drop(columns=['url', 'label']))

def is_phishing_url(url):
    """
    Loads the trained model and preprocessor to classify a single URL.
    Returns True if the URL is phishing, False otherwise.
    """
    # Load preprocessor and model
    with open('preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    model = load_model('phishguard.h5')

    # Prepare the input dataframe
    df = pd.DataFrame([[url, None]], columns=['url', 'label'])
    X = preprocessor.transform(df)

    # Make a prediction
    prediction = model.predict(X)

    # Return True if predicted phishing (below threshold)
    return prediction[0][0] < 0.5
