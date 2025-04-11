import os
import json
import pandas as pd
from nlp_engine import nlp_engine
from ml_model import ChatbotModel
from file_processor import clean_and_structure_data, extract_text

class Listener:
    def __init__(self, nlp_engine, model: ChatbotModel, data_dir='data'):
        self.nlp = nlp_engine
        self.model = model
        self.data_dir = data_dir
        self.stored_data = {}
        self.user_knowledge = {}
        os.makedirs(data_dir, exist_ok=True)
        self._load_stored_data()

    def _load_stored_data(self):
        data_file = os.path.join(self.data_dir, 'stored_data.json')
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                self.stored_data = json.load(f)

    def _save_stored_data(self):
        data_file = os.path.join(self.data_dir, 'stored_data.json')
        with open(data_file, 'w') as f:
            json.dump(self.stored_data, f)

    def learn_from_file(self, file_path: str):
        temp_path = os.path.join(self.data_dir, os.path.basename(file_path))
        structured_data = clean_and_structure_data(file_path, temp_path)
        if not structured_data:
            return "Failed to process file data"
        file_key = os.path.basename(file_path)
        self.stored_data[file_key] = structured_data
        self._save_stored_data()
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
        self.model.process_uploaded_data(file_content, file_key)
        return "Successfully learned from the file"

    def learn_from_user(self, question: str, answer: str):
        self.user_knowledge[question.lower()] = answer
        self.model.train([question, answer])
        self._save_stored_data()

    def handle_query(self, query: str) -> str:
        if "upload" in query.lower():
            return "Please upload the file using the UI"
        nlp_output = self.nlp(query)
        processed_query = nlp_output.get("raw_question", query.lower())
        response = self.model.predict([processed_query], nlp_output)[0]
        if "no relevant information found" in response.lower():
            listener_response = self._fallback_with_listener(query, nlp_output)
            if listener_response:
                self.learn_from_user(query, listener_response)
            return listener_response if listener_response else "No relevant information found"
        self.learn_from_user(query, response)
        return response

    def _fallback_with_listener(self, query: str, nlp_output: dict) -> str:
        query_lower = query.lower()
        intent = nlp_output.get("intent", "unknown")
        if intent == "total" and "amount" in query_lower:
            total = 0
            for file_data in self.stored_data.values():
                for row in file_data:
                    for value in row.values():
                        try:
                            total += float(str(value).replace(',', ''))
                        except ValueError:
                            continue
            return f"{total:.2f}" if total > 0 else "No numeric data found"
        target = nlp_output.get("target", "").lower()
        for key, value in self.user_knowledge.items():
            if key in query_lower:
                return value
        for file_data in self.stored_data.values():
            for row in file_data:
                row_text = str(row).lower()
                if target in row_text:
                    return str(row)  # Fallback to full row if not a total query
        return None