import streamlit as st
import re

class Listener:
    def __init__(self, model):
        self.model = model

    def listen(self, query, nlp_engine):
        cleaned_query = self._clean_text(query)
        if not self._check_data_presence(cleaned_query):
            listener_response = "Data not found in uploaded files"
            nlp_features = nlp_engine(query, listener_response)
            st.write(f"Listener: {listener_response}")
            return nlp_features
        return nlp_engine(query)

    def _clean_text(self, text):
        if not isinstance(text, str):
            text = str(text)
        text = text.lower().strip()
        text = re.sub(r'[^0-9a-zA-Z\s.-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def _check_data_presence(self, cleaned_query):
        for file_data in self.model.knowledge_base.values():
            for row in file_data['data']:
                row_text = ' '.join([str(v) for v in row.values()]).lower()
                if cleaned_query in row_text:
                    return True
        return False

if __name__ == "__main__":
    from ml_model import SmartChatbotModel
    from nlp_engine import nlp_engine
    model = SmartChatbotModel()
    listener = Listener(model)
    st.title("Smart Chatbot")
    st.write("Upload a CSV file and chat away. The bot learns from your file and user interactions!")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        model.process_uploaded_data(uploaded_file.getvalue().decode(), uploaded_file.name)
        st.success("Successfully learned from the file")
        query = st.text_input("Ask me anything!", key="unique_query")
        if st.button("Submit") or query:
            nlp_features = listener.listen(query, nlp_engine)
            response = model.predict([query], nlp_features)
            st.write(response[0])