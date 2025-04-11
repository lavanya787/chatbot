import nltk
import spacy

# Download required NLTK data (run once)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load SpaCy model (download if not present)
try:
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def extract_intent_and_entities(text):
    """
    Extract intent and entities from the input text using SpaCy.
    """
    doc = nlp(text.lower())
    intent = "unknown"
    entities = []
    
    # Determine intent based on key words or lemmas
    for token in doc:
        if token.lemma_ in ["list", "show", "display"]:
            intent = "list"
        elif token.lemma_ in ["total", "sum", "calculate"]:
            intent = "total"
        elif token.lemma_ in ["hello", "hi", "hey", "greet", "good", "morning", "afternoon", "evening"]:
            intent = "greeting"
    
    # Extract entities (named entities or context-specific terms)
    for ent in doc.ents:
        entities.append(ent.text.lower())
    
    # Add custom entities based on context
    if not entities and any(word in text.lower() for word in ["categories", "types", "labels"]):
        entities.append("categories")
    
    return intent, entities

def nlp_engine(question: str) -> dict:
    """
    Process the input question and return a dictionary of NLP features.
    """
    question = question.strip()
    intent, entities = extract_intent_and_entities(question)
    features, target = [], None
    
    # Extract features (noun chunks) and target
    doc = nlp(question.lower())
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.lower()
        if any(word in chunk_text for word in ["data", "information", "categories"]):
            target = chunk.text
        else:
            features.append(chunk.text)
    
    result = {
        "intent": intent,
        "entities": entities if entities else ["unknown"],
        "features": list(set(features)) if features else ["unknown"],
        "target": target if target else "unknown",
        "raw_question": question
    }
    return result

# Example usage (for testing)
if __name__ == "__main__":
    test_queries = ["hello", "Categories", "What is the total amount?", "What is the investment", "good morning"]
    for query in test_queries:
        result = nlp_engine(query)
        print(f"Query: {query}")
        print(f"NLP Output: {result}\n")