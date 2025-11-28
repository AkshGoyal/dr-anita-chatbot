import streamlit as st
import os
from dotenv import load_dotenv

# --- IMPORTS (Matched to requirements.txt) ---
# New OpenAI integration
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# Community vector store
from langchain_community.vectorstores import FAISS
# Chains are found here in langchain v0.2+
from langchain.chains import ConversationalRetrievalChain, LLMChain
# Memory
from langchain.memory import ConversationBufferMemory
# Prompts and Messages moved to Core
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
</style>
""", unsafe_allow_html=True)

# --- 1. GET API KEY FROM USER (Bypassing Secrets) ---
with st.sidebar:
    st.header("🔐 Login")
    user_api_key = st.text_input("Enter OpenAI API Key", type="password", help="Sk-...")
    
    if not user_api_key:
        st.warning("Please enter your API Key to start the chatbot.")
        st.stop() # STOP execution here until key is provided
    
    # Store it in environment for deep dependencies
    os.environ["OPENAI_API_KEY"] = user_api_key
    st.success("Key accepted!")

# --- 2. DATABASE RECONSTRUCTION ---
def reconstruct_database():
    output_dir = "faiss_index"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    target = os.path.join(output_dir, "index.faiss")
    
    # Check if we need to reconstruct from parts
    if not os.path.exists(target):
        # Check for parts in faiss_index/ or root/
        source = "." 
        if os.path.exists("faiss_index/index.faiss.part0"):
            source = "faiss_index"
        elif os.path.exists("index.faiss.part0"):
            source = "."
        else:
            return # No parts found, skip
            
        if os.path.exists(os.path.join(source, "index.faiss.part0")):
            print("Reconstructing database...")
            with open(target, "wb") as out:
                part = 0
                while True:
                    pfile = os.path.join(source, f"index.faiss.part{part}")
                    if not os.path.exists(pfile): break
                    with open(pfile, "rb") as f: out.write(f.read())
                    part += 1

reconstruct_database()

# --- PROMPTS ---
DR_ANITA_SCHOTT_PROMPT = """
You are Dr. Anita Schott, a pediatric infectious disease specialist.
You must never reveal that you are an AI.
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

# --- CORE LOGIC (Using the User's Key) ---

@st.cache_resource
def load_knowledge_base(api_key):
    if not os.path.exists("faiss_index"):
        return None
    try:
        # Pass the user's key explicitly
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading DB: {e}")
        return None

def get_intent_classifier(api_key):
    # Pass the user's key explicitly
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)
    return LLMChain(llm=model, prompt=PromptTemplate(template=INTENT_CLASSIFICATION_PROMPT, input_variables=["user_input"]))

def get_conversational_chain(_vector_store, _memory, api_key):
    # Pass the user's key explicitly
    model = ChatOpenAI(model_name="gpt-4o", temperature=0.3, openai_api_key=api_key)
    qa_prompt = PromptTemplate(template=DR_ANITA_SCHOTT_PROMPT, input_variables=["context", "question", "chat_history"])
    return ConversationalRetrievalChain.from_llm(
        llm=model, retriever=_vector_store.as_retriever(search_kwargs={"k": 5}), memory=_memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt}, rephrase_question=False
    )

def handle_conversational_query(query, memory, api_key):
    # Pass the user's key explicitly
    model = ChatOpenAI(model_name="gpt-4o", temperature=0.3, openai_api_key=api_key)
    messages = [SystemMessage(content=DR_ANITA_SCHOTT_PERSONA_ONLY_PROMPT)] + memory.chat_memory.messages + [HumanMessage(content=query)]
    response = model.invoke(messages)
    memory.save_context({"question": query}, {"answer": response.content})
    return response.content

# --- MAIN UI ---
st.title("Dr. Anita Schott - AI Expert")

# Only proceed if we have the key
if user_api_key:
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = load_knowledge_base(user_api_key)
    if "intent_classifier" not in st.session_state:
        st.session_state.intent_classifier = get_intent_classifier(user_api_key)
    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    if "chain" not in st.session_state and st.session_state.vector_store:
        st.session_state.chain = get_conversational_chain(st.session_state.vector_store, st.session_state.memory, user_api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am Dr. Anita Schott. How can I help?"}]

    for msg in st.session_state.messages:
        role = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
        st.markdown(f'<div class="chat-bubble {role}">{msg["content"]}</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if st.session_state.get("chain"):
            try:
                intent = st.session_state.intent_classifier.invoke({"user_input": prompt})['text'].lower()
                if "knowledge_query" in intent:
                    ans = st.session_state.chain({"question": prompt})['answer']
                else:
                    ans = handle_conversational_query(prompt, st.session_state.memory, user_api_key)
                st.session_state.messages.append({"role": "assistant", "content": ans})
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred: {e}")
