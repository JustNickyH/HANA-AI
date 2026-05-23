"""
main.py
HANA (Highly Adaptive Narrative Agent)
--------------------------------------
A sophisticated AI agent demonstrating LangChain architecture, semantic memory,
and emotion-adaptive conversational capabilities.
Core Features:
1. Emotion-Adaptive Tone: Analyzes user sentiment to adjust its conversational style.
2. Semantic Memory Retrieval: Uses a local ChromaDB vector database to recall past context.
Dependencies:
    pip install langchain langchain-openai langchain-chroma textblob
Usage:
    python main.py
"""
import os
import argparse
from typing import List, Dict, Any
from textblob import TextBlob
# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.documents import Document
# --- Configuration & Placeholders ---
# Replace with actual API keys or load from environment variables (.env)
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "sk-placeholder-openai-api-key")
class HANA:
    """
    Highly Adaptive Narrative Agent (HANA).
    
    This agent uses a local Chroma database to store and retrieve conversation history
    and adapts its conversational tone based on the user's sentiment.
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize HANA's brain, memory, and LLM components."""
        self.agent_name = "HANA"
        self.persist_directory = persist_directory
        
        # 1. Setup Embeddings and Vector Database (Semantic Memory)
        # Using Chroma as a lightweight local vector database
        self.embeddings = OpenAIEmbeddings()
        self.vector_db = Chroma(
            collection_name="hana_memory",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # 2. Setup the core Language Model
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.7)
        
        # 3. Short-term conversation history for context in current session
        self.chat_history: List[Any] = []
        
        # 4. Define the base prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", 
             "You are {agent_name}, a Highly Adaptive Narrative Agent. "
             "Your core directive is to be helpful, empathetic, and engaging. "
             "Adapt your tone according to the 'User Sentiment' provided. "
             "Use the 'Retrieved Context' from past interactions to inform your response and show that you remember the user.\n\n"
             "Retrieved Context:\n{context}\n\n"
             "User Sentiment: {sentiment}\n"
             "Tone Directive: {tone_directive}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze the user's input to determine sentiment and dictate the response tone.
        
        Args:
            text (str): The user's input text.
            
        Returns:
            dict: Contains the polarity score, sentiment label, and a tone directive.
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        
        if polarity > 0.3:
            sentiment = "Positive"
            directive = "Match the user's enthusiasm. Be upbeat, joyful, and encouraging."
        elif polarity < -0.3:
            sentiment = "Negative"
            directive = "Be highly empathetic, soothing, and supportive. Acknowledge their feelings gently."
        else:
            sentiment = "Neutral"
            directive = "Be polite, helpful, and objective. Maintain a friendly but balanced tone."
            
        return {
            "score": polarity,
            "label": sentiment,
            "directive": directive
        }
        
    def save_to_memory(self, user_input: str, ai_response: str):
        """
        Save the interaction to the semantic vector database.
        
        Args:
            user_input (str): What the user said.
            ai_response (str): What HANA replied.
        """
        # Create a document combining the interaction for context retrieval
        doc_content = f"User: {user_input}\nHANA: {ai_response}"
        doc = Document(page_content=doc_content)
        self.vector_db.add_documents([doc])
        
    def retrieve_memory(self, query: str, k: int = 3) -> str:
        """
        Retrieve relevant past interactions from the vector database.
        
        Args:
            query (str): The current user input to search against.
            k (int): Number of past interactions to retrieve.
            
        Returns:
            str: A formatted string of past interactions.
        """
        results = self.vector_db.similarity_search(query, k=k)
        if not results:
            return "No prior context found."
            
        context_chunks = [doc.page_content for doc in results]
        return "\n---\n".join(context_chunks)
        
    def process_message(self, user_input: str) -> str:
        """
        Core loop: analyze sentiment, retrieve memory, generate response, and save state.
        
        Args:
            user_input (str): The user's chat message.
            
        Returns:
            str: HANA's generated response.
        """
        # Step 1: Sentiment Analysis
        sentiment_analysis = self.analyze_sentiment(user_input)
        
        # Step 2: Semantic Memory Retrieval
        past_context = self.retrieve_memory(user_input)
        
        # Step 3: Format the prompt with all context
        chain = self.prompt_template | self.llm
        
        response = chain.invoke({
            "agent_name": self.agent_name,
            "context": past_context,
            "sentiment": sentiment_analysis["label"],
            "tone_directive": sentiment_analysis["directive"],
            "chat_history": self.chat_history,
            "input": user_input
        })
        
        ai_reply = str(response.content)
        
        # Step 4: Update short-term and long-term memory
        self.chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=ai_reply)
        ])
        
        self.save_to_memory(user_input, ai_reply)
        
        return ai_reply
def main():
    """
    CLI Interface for HANA.
    Provides a clean, interactive chat loop for the user.
    """
    print("=" * 60)
    print(" " * 20 + "PROJECT: HANA")
    print(" " * 10 + "Highly Adaptive Narrative Agent - Online")
    print("=" * 60)
    print("Type 'exit' or 'quit' to end the simulation.\n")
    
    # Initialize the agent
    print("[System]: Initializing HANA's neural pathways and memory...")
    try:
        hana = HANA()
        print("[System]: Initialization complete. HANA is ready.\n")
    except Exception as e:
        print(f"[Error]: Failed to initialize HANA. Details: {e}")
        print("Please ensure your OPENAI_API_KEY is set correctly.")
        return
    # Chat loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['exit', 'quit']:
                print("\nHANA: Goodbye! It was wonderful talking to you. My memory of this will persist.")
                break
                
            if not user_input:
                continue
                
            # Process and print response
            response = hana.process_message(user_input)
            print(f"\nHANA: {response}")
            
        except KeyboardInterrupt:
            print("\n\n[System]: Force shutdown initiated.")
            print("HANA: Goodbye!")
            break
        except Exception as e:
            print(f"\n[System Error]: {e}\n")
if __name__ == "__main__":
    main()
