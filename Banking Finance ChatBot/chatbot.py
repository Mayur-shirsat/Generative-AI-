import cohere
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import tkinter as tk
from tkinter import scrolledtext

# Initialize Cohere client with the API key
API_KEY = "pYUL24Qf3UJPCqForAQWP8ABzh2wKJzKjhm0pfH4"
cohere_client = cohere.Client(API_KEY)

# Load the dataset and generate embeddings
data = pd.read_csv("BankFAQs.csv")

def generate_embeddings(data, cohere_client):
    """Generate embeddings for the questions in the dataset using Cohere."""
    questions = data['Question'].tolist()
    response = cohere_client.embed(texts=questions, model="embed-english-v2.0")
    data['Embedding'] = response.embeddings
    return data

# Precompute embeddings for all dataset questions
data = generate_embeddings(data, cohere_client)

# Build a KDTree for efficient nearest-neighbor search
embeddings_array = np.array(data['Embedding'].tolist())
embedding_tree = KDTree(embeddings_array)

# Create a hash map for quick category-based filtering
category_map = {category: data[data['Class'] == category] for category in data['Class'].unique()}

def get_closest_question(user_query, cohere_client):
    """Find the most relevant question in the dataset using K-D Tree."""
    query_embedding = cohere_client.embed(texts=[user_query], model="embed-english-v2.0").embeddings[0]
    dist, idx = embedding_tree.query([query_embedding], k=1)
    best_match_idx = idx[0][0]
    closest_question = data.iloc[best_match_idx]['Question']
    answer = data.iloc[best_match_idx]['Answer']
    return closest_question, answer

def classify_query(user_query, cohere_client):
    """Classify the user query into one of the predefined categories."""
    examples = [{"text": row['Question'], "label": row['Class']} for _, row in data.iterrows()]
    response = cohere_client.classify(inputs=[user_query], examples=examples, model="large")
    return response.classifications[0].prediction

def chatbot(conversation_history, user_query):
    """Process user input, classify it, and find the best matching answer."""
    try:
        predicted_class = classify_query(user_query, cohere_client)
        filtered_data = category_map[predicted_class]
        filtered_embeddings = np.array(filtered_data['Embedding'].tolist())
        filtered_tree = KDTree(filtered_embeddings)
        
        query_embedding = cohere_client.embed(texts=[user_query], model="embed-english-v2.0").embeddings[0]
        dist, idx = filtered_tree.query([query_embedding], k=1)
        best_match_idx = idx[0][0]
        
        closest_question = filtered_data.iloc[best_match_idx]['Question']
        answer = filtered_data.iloc[best_match_idx]['Answer']
        
        conversation_history.append(("User", user_query))
        conversation_history.append(("Bot", f"**Closest Question:** {closest_question}\n\n**Answer:** {answer}"))

        chat_transcript = "\n\n".join([f"{speaker}: {message}" for speaker, message in conversation_history])
        return chat_transcript, conversation_history
    except Exception as e:
        conversation_history.append(("Bot", f"An error occurred: {e}"))
        chat_transcript = "\n\n".join([f"{speaker}: {message}" for speaker, message in conversation_history])
        return chat_transcript, conversation_history

class ChatbotApp:
    def __init__(self, root):
        """Initialize the Tkinter GUI for the chatbot."""
        self.root = root
        self.root.title("AI Chatbot with Cohere")
        self.root.geometry("375x667")  # Mobile screen size (width x height)
        self.root.configure(bg="#f4f4f9")  # Background color of the window
        self.conversation_history = []

        # Create UI components
        self.chat_window = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=15, width=40, state="disabled", font=("Arial", 10), bg="#ffffff", fg="#333333", bd=2, relief="groove")
        self.chat_window.grid(row=0, column=0, padx=10, pady=10, columnspan=2)

        self.entry = tk.Entry(root, width=30, font=("Arial", 10), bd=2, relief="sunken", fg="#333333", bg="#f0f0f0")
        self.entry.grid(row=1, column=0, padx=10, pady=10)
        self.entry.bind("<Return>", self.process_user_input)

        self.send_button = tk.Button(root, text="Send", font=("Arial", 10, "bold"), command=self.process_user_input, bg="#4CAF50", fg="white", bd=2, relief="raised")
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        # Display initial greeting message when app starts
        self.initial_greeting()

    def initial_greeting(self):
        """Display a welcome message when the chatbot window is first opened."""
        greeting_message = "Hi, my name is Nemo, an assistant for you. How can I help you?"
        self.chat_window.configure(state="normal")
        self.chat_window.insert(tk.END, "Bot: " + greeting_message + "\n\n")
        self.chat_window.configure(state="disabled")
        self.chat_window.yview(tk.END)  # Auto-scroll to the bottom

    def process_user_input(self, event=None):
        """Process the user input, generate response and update the chat window."""
        user_query = self.entry.get()
        if not user_query.strip():
            return

        self.entry.delete(0, tk.END)

        # Get response from the chatbot
        chat_transcript, self.conversation_history = chatbot(self.conversation_history, user_query)

        # Update chat window
        self.chat_window.configure(state="normal")
        self.chat_window.delete(1.0, tk.END)  # Clear the previous chat
        self.chat_window.insert(tk.END, chat_transcript)  # Insert new conversation
        self.chat_window.configure(state="disabled")  # Disable editing
        self.chat_window.yview(tk.END)  # Auto-scroll to the bottom

# Initialize and run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = ChatbotApp(root)
    root.mainloop()
