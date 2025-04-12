import streamlit as st
from nlp_engine import NLPEngine
from ml_model import ChatbotModel
from listener import Listener
import tempfile

st.title("📚 Smart Chatbot with File Learning")

nlp_engine = NLPEngine()
chatbot_model = ChatbotModel()
listener = Listener(nlp_engine, chatbot_model)

# Load trained model if available
try:
    chatbot_model.load("models/chatbot_model.pkl")
except:
    pass

uploaded_file = st.file_uploader("Upload a file (PDF, CSV, XLSX)", type=["pdf", "csv", "xlsx"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name

    msg = listener.learn_from_file(tmp_file_path)
    st.success(msg)

user_input = st.text_input("Ask something:")
if st.button("Send") and user_input:
    response = listener.handle_query(user_input)
    st.write("Bot:", response)
