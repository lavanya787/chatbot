import streamlit as st
from listener import Listener
from nlp_engine import nlp_engine
from ml_model import ChatbotModel
from file_processor import clean_and_structure_data
import os

def main():
    model = ChatbotModel()
    listener = Listener(nlp_engine, model)
    st.title("Smart Chatbot")
    st.write("Upload a file and chat away. The bot learns from your file and user interactions!")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        file_path = os.path.join("data", f"temp_{uploaded_file.name}")
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        result = listener.learn_from_file(file_path)
        st.success(result)
    query = st.text_input("Ask me anything!")
    if query:
        response = listener.handle_query(query)
        st.write(response)

if __name__ == "__main__":
    main()