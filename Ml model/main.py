import streamlit as st
from ml_model import SmartChatbotModel
from listener import Listener
from nlp_engine import nlp_engine

def main():
    st.title("Smart Chatbot")
    st.write("Upload a file and chat away. The bot learns from your file and user interactions!")

    model = SmartChatbotModel()
    listener = Listener(model)

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file:
        model.process_uploaded_data(uploaded_file.getvalue().decode(), uploaded_file.name)
        st.success("Successfully learned from the file")
        query = st.text_input("Ask me anything!", key="unique_query")
        if st.button("Submit") or query:
            nlp_features = listener.listen(query, nlp_engine)
            response = model.predict([query], nlp_features)
            st.write(response[0])

if __name__ == "__main__":
    main()