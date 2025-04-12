import os
import pandas as pd
import pickle
import re
import time
from io import StringIO
from difflib import get_close_matches

class SmartChatbotModel:
    def __init__(self, model_path='models/smart_chatbot_model.pkl', data_dir='data'):
        self.model_path = model_path
        self.data_dir = data_dir
        self.knowledge_base = {}
        self.trained = False
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.knowledge_base = model_data.get('knowledge_base', {})
                self.trained = bool(self.knowledge_base)

    def save(self):
        with open(self.model_path, 'wb') as f:
            pickle.dump({'knowledge_base': self.knowledge_base}, f)

    def _clean_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        return re.sub(r'\s+', ' ', text.strip().lower())

    def _is_numeric(self, value):
        try:
            float(str(value).replace(',', '').replace('$', ''))
            return True
        except ValueError:
            return False

    def process_uploaded_data(self, file_content, file_name):
        try:
            if file_name.lower().endswith('.csv'):
                df = pd.read_csv(StringIO(file_content))
                df.dropna(inplace=True)
                
                # Vectorized cleaning: Convert all values to string and strip them
                df = df.apply(lambda x: x.astype(str).str.strip())
                
                records = df.to_dict(orient='records')
                metadata = {col: str(df[col].dtype) for col in df.columns}

                # Store data for each file separately and track each file's metadata
                self.knowledge_base[file_name] = {
                    'data': records,
                    'metadata': metadata,
                    'timestamp': time.time()
                }

                # Indicate that the model has been trained
                self.trained = True
                self.save()
                return f"Successfully learned from the file: {file_name}"
            return "Unsupported file format"
        except Exception as e:
            return f"[ERROR] Failed to process file: {str(e)}"

    def predict(self, query, nlp_features=None):
        if not self.trained:
            return ["Model not trained. Upload data to get started."]

        cleaned_query = self._clean_text(query)
        target_field = self._match_field(cleaned_query)

        greetings = ["hi", "hello", "hey"]
        if any(greet in cleaned_query for greet in greetings):
            return ["👋 Hello! How can I help you with your data?"]

        if not target_field:
            all_fields = self.get_all_fields()
            return [
                "🤖 Listener: I couldn't find a relevant match for your query in the uploaded data.",
                f"📌 Available fields: {', '.join(sorted(all_fields))}"
            ]

        return [self._generate_response(cleaned_query, target_field)]

    def _match_field(self, query):
        all_fields = self.get_all_fields()
        query_words = query.split()

        for field in all_fields:
            if field in query:
                return field

        for word in query_words:
            matches = get_close_matches(word, all_fields, n=1, cutoff=0.8)
            if matches:
                return matches[0]

        return None

    def get_all_fields(self):
        fields = set()
        for file_data in self.knowledge_base.values():
            fields.update([self._clean_text(f) for f in file_data['metadata'].keys()])
        return list(fields)

    def _generate_response(self, query, field):
        all_values = []
        is_numeric = True

        # Iterate through all data from all files
        for file_data in self.knowledge_base.values():
            for row in file_data['data']:
                for col, val in row.items():
                    if self._clean_text(col) == field:
                        all_values.append(val)
                        if not self._is_numeric(val):
                            is_numeric = False

        if not all_values:
            return f"No values found for '{field}'"

        numeric_values = []
        if is_numeric:
            numeric_values = [float(str(v).replace(',', '').replace('$', '')) for v in all_values]

        if any(keyword in query for keyword in ["average", "mean"]):
            if is_numeric:
                return f"📊 Average value of '{field}': {sum(numeric_values)/len(numeric_values):.2f}"
            return f"⚠️ Cannot compute average for non-numeric field '{field}'"

        if any(keyword in query for keyword in ["total", "sum"]):
            if is_numeric:
                return f"💰 Total sum of '{field}': {sum(numeric_values):.2f}"
            return f"⚠️ Cannot compute total for non-numeric field '{field}'"

        if any(keyword in query for keyword in ["maximum", "max", "highest"]):
            if is_numeric:
                return f"🔺 Maximum value of '{field}': {max(numeric_values):.2f}"
            return f"⚠️ Cannot find maximum for non-numeric field '{field}'"

        if any(keyword in query for keyword in ["minimum", "min", "lowest"]):
            if is_numeric:
                return f"🔻 Minimum value of '{field}': {min(numeric_values):.2f}"
            return f"⚠️ Cannot find minimum for non-numeric field '{field}'"

        if any(keyword in query for keyword in ["list", "show", "types", "what", "values"]):
            unique = sorted(set(str(v) for v in all_values))
            return f"📋 Unique values for '{field}': {', '.join(unique)}"

        return f"📌 Values found for '{field}' — Try asking for list, total, average, maximum, or minimum!"
