import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np
from scipy.sparse import vstack
import pandas as pd
from io import StringIO
import re
import time

class ChatbotModel:
    def __init__(self, model_path='models/chatbot_model.pkl', data_dir='data'):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),  # Include bigrams for better context
            min_df=1
        )
        self.corpus = []
        self.tfidf_matrix = None
        self.trained = False
        self.model_path = model_path
        self.data_dir = data_dir
        self.stored_data = {}  # Store raw structured data
        self._load_model()
        os.makedirs(self.data_dir, exist_ok=True)

    def _load_model(self):
        start_time = time.time()
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.vectorizer = data['vectorizer']
                    self.corpus = data['corpus']
                    self.tfidf_matrix = data['tfidf_matrix']
                    self.trained = True
                    if 'stored_data' in data:
                        self.stored_data = data['stored_data']
            except Exception as e:
                self.trained = False
        print(f"Model loaded in {time.time() - start_time:.3f} seconds")

    def _clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def process_uploaded_data(self, file_content: str, file_name: str):
        start_time = time.time()
        try:
            texts = []
            if file_name.endswith('.csv'):
                df = pd.read_csv(StringIO(file_content), low_memory=False)
                self.stored_data[file_name] = df.to_dict('records')  # Store all data
                # Enrich training data with structured key-value pairs and context
                for index, row in df.iterrows():
                    text = ' '.join([f"{k} {v}" for k, v in row.items()])
                    # Add context by including combinations of fields
                    for i in range(len(row)):
                        for j in range(i + 1, len(row)):
                            keys = list(row.index)[i:j+1]
                            values = [row[k] for k in keys]
                            text += ' ' + ' '.join([f"{k} {v}" for k, v in zip(keys, values)])
                    texts.append(text)
            else:
                texts = [line.strip() for line in file_content.split('\n') if line.strip()]

            cleaned_texts = [self._clean_text(text) for text in texts if text]
            cleaned_texts = [t for t in cleaned_texts if len(t) > 5]

            if cleaned_texts:
                self.train(cleaned_texts)
                save_path = os.path.join(self.data_dir, f"processed_{file_name}")
                with open(save_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(cleaned_texts))
                print(f"Processed {len(cleaned_texts)} entries from {file_name} in {time.time() - start_time:.3f} seconds")
            else:
                print("No valid data found in upload")
        except Exception as e:
            print(f"Error processing uploaded data: {e}")

    def train(self, texts: list):
        start_time = time.time()
        if not texts:
            return
        cleaned_texts = list(dict.fromkeys([self._clean_text(t) for t in texts if t]))
        new_texts = [t for t in cleaned_texts if t not in self.corpus]
        if not new_texts:
            return

        self.corpus.extend(new_texts)
        new_tfidf = self.vectorizer.fit_transform(new_texts) if not self.trained else self.vectorizer.transform(new_texts)
        self.tfidf_matrix = new_tfidf if self.tfidf_matrix is None else vstack([self.tfidf_matrix, new_tfidf])
        self.trained = True
        self.save()
        print(f"Trained on {len(new_texts)} new samples in {time.time() - start_time:.3f} seconds")

    def predict(self, query: list, nlp_features: dict = None):
        start_time = time.time()
        print(f"Query: {query}, NLP features: {nlp_features}")
        if not self.trained or not self.corpus:
            print("Model not trained")
            return ["Model not trained yet"]

        cleaned_query = [self._clean_text(q) for q in query if q]
        if not cleaned_query:
            return ["Invalid query"]

        query_tfidf = self.vectorizer.transform(cleaned_query)
        similarity = cosine_similarity(query_tfidf, self.tfidf_matrix)
        threshold = 0.1

        if similarity.max() < threshold:
            print("No strong match, checking structured data")
            response = self._handle_structured_query(nlp_features, cleaned_query)
            if response:
                print(f"Structured response: {response} (time: {time.time() - start_time:.3f} seconds)")
                return [response]
            print("No relevant match found")

        if similarity.max() < threshold:
            return ["No relevant information found"]

        best_idx = np.argmax(similarity[0])
        response = self.corpus[best_idx]
        print(f"Response: {response} (time: {time.time() - start_time:.3f} seconds)")
        return [response]

    def _handle_structured_query(self, nlp_features: dict, cleaned_query: list):
        intent = nlp_features.get("intent", "unknown")
        entities = nlp_features.get("entities", ["unknown"])
        target = nlp_features.get("target", "").lower()

        if intent == "greeting":
            return "Hello! How can I assist you today?"

        elif intent == "list" and "categories" in entities:
            if self.stored_data:
                all_categories = set()
                for file_data in self.stored_data.values():
                    for row in file_data:
                        for value in row.values():
                            # Include only non-numeric strings as categories
                            if isinstance(value, str) and not value.replace('.', '').replace('-', '').isdigit():
                                all_categories.add(value)
                unique_categories = sorted(list(all_categories))
                return f"Categories: {', '.join(unique_categories)}" if unique_categories else "No categories found"

        elif intent == "total":
            if self.stored_data:
                total = 0
                for file_data in self.stored_data.values():
                    for row in file_data:
                        for value in row.values():
                            try:
                                total += float(str(value).replace(',', ''))
                            except ValueError:
                                continue
                return f"{total:.2f}" if total > 0 else "No numeric data found"

        elif target and any(e in ["category", "type"] for e in entities):
            results = []
            if self.stored_data:
                for file_data in self.stored_data.values():
                    for row in file_data:
                        row_text = ' '.join([f"{k} {v}" for k, v in row.items()]).lower()
                        if target in row_text or any(q in row_text for q in cleaned_query):
                            results.append(str(row))
                unique_results = sorted(list(set(results)), key=lambda x: x)
                return '; '.join(unique_results) if unique_results else "No matching categories found"

        return None

    def _check_uploaded_data(self, query: list):
        start_time = time.time()
        try:
            for file_name in os.listdir(self.data_dir):
                file_path = os.path.join(self.data_dir, file_name)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                if any(q.lower() in content for q in query):
                    self.process_uploaded_data(content, file_name)
                    print(f"Retrained from {file_name} in {time.time() - start_time:.3f} seconds")
                    break
        except Exception as e:
            print(f"Error checking uploaded data: {e}")

    def save(self):
        start_time = time.time()
        try:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'corpus': self.corpus,
                    'tfidf_matrix': self.tfidf_matrix,
                    'stored_data': self.stored_data
                }, f)
            print(f"Model saved in {time.time() - start_time:.3f} seconds")
        except Exception as e:
            print(f"Error saving model: {e}")