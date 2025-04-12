import nltk
import spacy

try:
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nlp = spacy.load("en_core_web_sm")
except:
    from spacy.cli import download
    download("en_core_web_sm")
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nlp = spacy.load("en_core_web_sm")

def extract_intent_and_entities(text):
    doc = nlp(text.lower())
    intent = "unknown"
    entities = []
    
    for token in doc:
        if token.lemma_ in ["list", "show", "display"]:
            intent = "list"
        elif token.lemma_ in ["total", "sum", "calculate", "average"]:
            intent = "total"
        elif token.lemma_ in ["hello", "hi", "hey", "greet", "good", "morning", "afternoon", "evening"]:
            intent = "greeting"
        elif token.lemma_ in ["no", "wrong", "incorrect", "retry"] and any(word in text.lower() for word in ["again", "recheck", "correct"]):
            intent = "correct"

    for ent in doc.ents:
        entities.append(ent.text.lower())
    
    if not entities and any(word in text.lower() for word in ["categories", "types", "labels"]):
        entities.append("categories")
    elif not entities and any(word in text.lower() for word in ["sum", "total", "average"]):
        entities.append("total")

    return intent, entities

def nlp_engine(question, listener_response=None):
    question = question.strip()
    intent, entities = extract_intent_and_entities(question)
    features, target = [], None
    
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
        "raw_question": question,
        "listener_feedback": listener_response if listener_response else None
    }
    print(f"NLP Output: {result}")
    return result

if __name__ == "__main__":
    test_queries = ["hello", "Categories", "What is the total", "What is the withdrawal", "No, what is the withdrawal?"]
    for query in test_queries:
        result = nlp_engine(query)
        print(f"Query: {query}")
        print(f"NLP Output: {result}\n")