import streamlit as st
import os
from dotenv import load_dotenv

# --- MODERN IMPORTS (Compatible with latest langchain) ---
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
# These moved to langchain_core in the new versions
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# ---------------------------------------------------------

# --- PAGE CONFIG ---
st.set_page_config(page_title="Dr. Anita Schott - AI Medical Expert", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    .chat-bubble { border-radius: 10px; padding: 16px; margin-bottom: 12px; border: 1px solid #e0e0e0; max-width: 80%; }
    .user-bubble { background-color: #d0f0fd; margin-left: auto; }
    .assistant-bubble { background-color: #ffffff; }
    .stButton>button { background-color: #e0e0e0; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 1. GET API KEY FROM USER (Manual Input + Sanitizer) ---
with st.sidebar:
    st.header("🔐 Login")
    raw_key = st.text_input("Enter OpenAI API Key", type="password", help="Starts with sk-proj...")
    
    if not raw_key:
        st.warning("Please enter your API Key to start the chatbot.")
        st.stop() # STOP execution here until key is provided
    
    # SANITIZER: Automatically remove accidental spaces, newlines, or quotes
    user_api_key = raw_key.strip().replace("'", "").replace('"', "")
    
    # Set to environment (backup)
    os.environ["OPENAI_API_KEY"] = user_api_key
    st.success("Key accepted!")

# --- 2. DATABASE RECONSTRUCTION ---
def reconstruct_database():
    output_dir = "faiss_index"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    target = os.path.join(output_dir, "index.faiss")
    
    # Check if we need to reconstruct (if target is missing)
    if not os.path.exists(target):
        source = "." 
        # Check if parts are in the folder or root
        if os.path.exists("faiss_index/index.faiss.part0"):
            source = "faiss_index"
        elif os.path.exists("index.faiss.part0"):
            source = "."
        else:
            return # No parts found
            
        print("Reconstructing database from parts...")
        with open(target, "wb") as out:
            part = 0
            while True:
                pfile = os.path.join(source, f"index.faiss.part{part}")
                if not os.path.exists(pfile): break
                with open(pfile, "rb") as f: out.write(f.read())
                part += 1
        print("Database reconstructed successfully.")

reconstruct_database()

# --- PROMPTS ---
DR_ANITA_SCHOTT_PROMPT = """
You are Dr. Anita Schott, a pediatric infectious disease specialist.
You must never reveal that you are an AI. You are a human doctor.

## MANDATORY LOOP
1. Analyze User Query.
2. Triage:
    A) Factual? Answer directly.
    B) Recommendation? Check chat_history for patient age/location. 
       - If missing: Ask clarifying questions.
       - If present: Provide recommendation.

Context: {context}
History: {chat_history}
Question: {question}
Answer:
"""

DR_ANITA_SCHOTT_PERSONA_ONLY_PROMPT = """
You are Dr. Anita Schott. Respond to small talk naturally and professionally.
"""

INTENT_CLASSIFICATION_PROMPT = """
Categorize the query into: 1. knowledge_query, 2. conversational_query
User Query: "{user_input}"
Intent:
"""

# --- CORE LOGIC (Using Explicit API Key) ---

@st.cache_resource
def load_knowledge_base(api_key):
    if not os.path.exists("faiss_index"):
        return None
    try:
        # Explicitly pass the key to embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading Knowledge Base. Details: {e}")
        return None

def get_intent_classifier(api_key):
    try:
        # Explicitly pass the key
        model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
        return LLMChain(llm=model, prompt=PromptTemplate(template=INTENT_CLASSIFICATION_PROMPT, input_variables=["user_input"]))
    except Exception as e:
        st.error(f"Error initializing Classifier: {e}")
        st.stop()

def get_conversational_chain(_vector_store, _memory, api_key):
    try:
        # Explicitly pass the key
        model = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key=api_key)
        qa_prompt = PromptTemplate(template=DR_ANITA_SCHOTT_PROMPT, input_variables=["context", "question", "chat_history"])
        return ConversationalRetrievalChain.from_llm(
            llm=model, retriever=_vector_store.as_retriever(search_kwargs={"k": 5}), memory=_memory,
            combine_docs_chain_kwargs={"prompt": qa_prompt}, rephrase_question=False
        )
    except Exception as e:
        st.error(f"Error initializing Chat Chain: {e}")
        st.stop()

def handle_conversational_query(query, memory, api_key):
    # Explicitly pass the key
    model = ChatOpenAI(model="gpt-4o", temperature=0.3, openai_api_key=api_key)
    messages = [SystemMessage(content=DR_ANITA_SCHOTT_PERSONA_ONLY_PROMPT)] + memory.chat_memory.messages + [HumanMessage(content=query)]
    response = model.invoke(messages)
    memory.save_context({"question": query}, {"answer": response.content})
    return response.content

# --- MAIN UI ---
st.title("Dr. Anita Schott - AI Expert")

if user_api_key:
    # Initialize components with the manual key
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = load_knowledge_base(user_api_key)
    
    if "intent_classifier" not in st.session_state:
        st.session_state.intent_classifier = get_intent_classifier(user_api_key)
        
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
        
    if "chain" not in st.session_state and st.session_state.vector_store:
        st.session_state.chain = get_conversational_chain(st.session_state.vector_store, st.session_state.memory, user_api_key)

    # Chat Interface
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am Dr. Anita Schott. How can I help?"}]

    for msg in st.session_state.messages:
        role = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
        st.markdown(f'<div class="chat-bubble {role}">{msg["content"]}</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Only process if chain exists
        if st.session_state.get("chain"):
            try:
                # 1. Classify
                intent = st.session_state.intent_classifier.invoke({"user_input": prompt})['text'].lower()
                
                # 2. Route
                if "knowledge_query" in intent:
                    ans = st.session_state.chain({"question": prompt})['answer']
                else:
                    ans = handle_conversational_query(prompt, st.session_state.memory, user_api_key)
                
                st.session_state.messages.append({"role": "assistant", "content": ans})
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")
        else:
            st.error("Knowledge base not loaded. Please ensure PDFs were ingested and files uploaded.")
