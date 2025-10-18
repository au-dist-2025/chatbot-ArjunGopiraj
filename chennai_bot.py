import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama

with open("chennai_knowledge.json", "r") as file:
    documents = json.load(file)

doc_texts = list(documents.values())
doc_keys = list(documents.keys())

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embed_model.encode(doc_texts)

def retrieve_context(query, top_k=2):
    query_embedding = embed_model.encode([query])
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [doc_texts[i] for i in top_indices]

def rag_llm_chatbot(query, user_name=None):
    context = retrieve_context(query)
    prompt = f"""
    You are a Chennai Enquiry Bot for tourists and locals.
    Use the following context to answer naturally:

    Context: {" ".join(context)}
    User Query: {query}
    """
    response = ollama.chat(
        model="mistral",
        messages=[
            {"role": "system", "content": "You are a helpful Chennai tourist assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    answer = response['message']['content']
    if user_name and "hi" in query.lower():
        answer += f" Nice to see you again, {user_name}!"
    return answer

def chat():
    print("CHENNAI Enquiry Bot: Hello! Ask me about places, food, shopping, movies, or culture.")
    user_name = None
    while True:
        query = input("You: ")
        if "bye" in query.lower():
            print("Bot: Goodbye! Have a great time in Chennai")
            break
        elif "my name is" in query.lower():
            user_name = query.split("is")[-1].strip().capitalize()
            print(f"Bot: Nice to meet you, {user_name}! I'll remember your name.")
            continue
        elif "what's my name" in query.lower():
            if user_name:
                print(f"Bot: Your name is {user_name}.")
            else:
                print("Bot: I don't know your name yet. Tell me using 'My name is ...'")
            continue
        response = rag_llm_chatbot(query, user_name)
        print("Bot:", response)

if __name__ == "__main__":
    chat()
