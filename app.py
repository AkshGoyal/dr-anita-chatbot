import streamlit as st
import os
from dotenv import load_dotenv

# --- STABLE IMPORTS (Compatible with langchain==0.1.20) ---
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage

# --- 1. BOOTSTRAP: Secure Key Loader ---
# This function guarantees we get a clean key or stop immediately.
def get_clean_api_key():
    # Check if key exists in secrets
    if "OPENAI_API_KEY" not in st.secrets:
        st.error("🚨 CRITICAL ERROR: OpenAI API Key is missing from Secrets!")
        st.info("Go to 'Manage App' -> 'Settings' -> 'Secrets' and add it.")
        st.stop()
    
    # Get the raw key
    raw_key = st.secrets["OPENAI_API_KEY"]
    
    # CLEAN IT: Remove accidental spaces, newlines, or quote marks
    clean_key = raw_key.strip().replace("'", "").replace('"', "")
    
    return clean_key

# Get the key ONCE and use it everywhere
API_KEY = get_clean_api_key()

# Set it to environment just in case, but we will pass it explicitly too
os.environ["OPENAI_API_KEY"] = API_KEY
# ---------------------------------------

# --- App Configuration ---
st.set_page_config(page_title="Dr. Anita Schott - AI Medical Expert", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    .chat-bubble { border-radius: 10px; padding: 16px; margin-bottom: 12px; border: 1px solid #e0e0e0; max-width: 80%; }
    .user-bubble { background-color: #d0f0fd; margin-left: auto; }
    .assistant-bubble { background-color: #ffffff; }
</style>
""", unsafe_allow_html=True)


# --- Prompts (Unchanged) ---
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


# --- Core Logic with EXPLICIT Key Passing ---

def reconstruct_database():
    output_dir = "faiss_index"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    target = os.path.join(output_dir, "index.faiss")
    
    if not os.path.exists(target):
        # Only try to reconstruct if parts exist
        if os.path.exists(f"{target}.part0") or os.path.exists(f"{output_dir}/{target}.part0"):
            print("Reconstructing database...")
            source = output_dir if os.path.exists(os.path.join(output_dir, "index.faiss.part0")) else "."
            with open(target, "wb") as out:
                part = 0
                while True:
                    pfile = os.path.join(source, f"index.faiss.part{part}")
                    if not os.path.exists(pfile): break
                    with open(pfile, "rb") as f: out.write(f.read())
                    part += 1
            print("Database reconstructed.")

reconstruct_database()

@st.cache_resource
def load_knowledge_base():
    if not os.path.exists("faiss_index"):
        st.sidebar.error("Database folder missing.")
        return None
    try:
        # EXPLICIT KEY PASSING #1
        embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        return None

@st.cache_resource
def get_intent_classifier():
    # EXPLICIT KEY PASSING #2
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=API_KEY)
    return LLMChain(llm=model, prompt=PromptTemplate(template=INTENT_CLASSIFICATION_PROMPT, input_variables=["user_input"]))

def get_conversational_chain(_vector_store, _memory):
    # EXPLICIT KEY PASSING #3
    model = ChatOpenAI(model_name="gpt-4o", temperature=0.3, openai_api_key=API_KEY)
    qa_prompt = PromptTemplate(template=DR_ANITA_SCHOTT_PROMPT, input_variables=["context", "question", "chat_history"])
    return ConversationalRetrievalChain.from_llm(
        llm=model, retriever=_vector_store.as_retriever(search_kwargs={"k": 5}), memory=_memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt}, rephrase_question=False
    )

def handle_conversational_query(query, memory):
    # EXPLICIT KEY PASSING #4
    model = ChatOpenAI(model_name="gpt-4o", temperature=0.3, openai_api_key=API_KEY)
    messages = [SystemMessage(content=DR_ANITA_SCHOTT_PERSONA_ONLY_PROMPT)] + memory.chat_memory.messages + [HumanMessage(content=query)]
    response = model.invoke(messages)
    memory.save_context({"question": query}, {"answer": response.content})
    return response.content

# --- Main UI ---
st.title("Dr. Anita Schott - AI Expert")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_knowledge_base()
if "intent_classifier" not in st.session_state:
    st.session_state.intent_classifier = get_intent_classifier()
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
if "chain" not in st.session_state and st.session_state.vector_store:
    st.session_state.chain = get_conversational_chain(st.session_state.vector_store, st.session_state.memory)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am Dr. Anita Schott. How can I help?"}]

for msg in st.session_state.messages:
    role = "user-bubble" if msg["role"] == "user" else "assistant-bubble"
    st.markdown(f'<div class="chat-bubble {role}">{msg["content"]}</div>', unsafe_allow_html=True)

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    if st.session_state.get("chain"):
        intent = st.session_state.intent_classifier.invoke({"user_input": prompt})['text'].lower()
        if "knowledge_query" in intent:
            ans = st.session_state.chain({"question": prompt})['answer']
        else:
            ans = handle_conversational_query(prompt, st.session_state.memory)
        st.session_state.messages.append({"role": "assistant", "content": ans})
        st.rerun()
