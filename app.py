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
# ---------------------------------------------------------

# Load environment variables (local dev)
load_dotenv()

# --- KEY SANITIZER (The Fix) ---
# This block cleans the key from secrets before using it
if "OPENAI_API_KEY" in st.secrets:
    raw_key = st.secrets["OPENAI_API_KEY"]
    # 1. Remove spaces/newlines
    clean_key = raw_key.strip()
    # 2. Remove accidental quotes inside the string
    clean_key = clean_key.replace('"', '').replace("'", "")
    
    # 3. Set the clean key to the environment
    os.environ["OPENAI_API_KEY"] = clean_key
    
    # Debug: Show us if it worked (Safe - only shows length)
    print(f"DEBUG: API Key loaded. Length: {len(clean_key)}")
else:
    st.error("🚨 OpenAI API Key is missing! Please check 'Secrets' tab.")
    st.stop()
# -------------------------------

# --- SMART GLUE CODE: Reconstruct the heavy file ---
def reconstruct_database():
    output_dir = "faiss_index"
    output_file = os.path.join(output_dir, "index.faiss")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_file):
        print("Database already exists. Skipping reconstruction.")
        return

    part_prefix = "index.faiss.part"
    if os.path.exists(os.path.join(output_dir, f"{part_prefix}0")):
        source_dir = output_dir
    elif os.path.exists(f"{part_prefix}0"):
        source_dir = "."
    else:
        # If parts are missing, we can't reconstruct, but we don't want to crash 
        # immediately if the user hasn't uploaded them yet.
        print("Warning: Database parts not found. Skipping reconstruction.")
        return

    print(f"Reconstructing database from parts found in '{source_dir}'...")
    
    with open(output_file, "wb") as outfile:
        part_num = 0
        while True:
            part_path = os.path.join(source_dir, f"{part_prefix}{part_num}")
            if not os.path.exists(part_path):
                break
            with open(part_path, "rb") as partfile:
                outfile.write(partfile.read())
            part_num += 1
            
    print(f"Success! Database reconstructed at {output_file}")

# Run reconstruction
reconstruct_database()


# --- App Configuration & Styling ---
st.set_page_config(page_title="Dr. Anita Schott - AI Medical Expert", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f0f2f6; }
    h1 { text-align: center; font-size: 2rem; margin-top: -40px; padding-top: 0px; }
    .chat-bubble { border-radius: 10px; padding: 16px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #e0e0e0; width: fit-content; max-width: 75%; word-wrap: break-word; }
    .user-bubble { background-color: #d0f0fd; margin-left: auto; }
    .assistant-bubble { background-color: #ffffff; align-self: flex-start; }
    .stButton>button { background-color: #e0e0e0; color: #333; border: none; border-radius: 10px; padding: 8px 12px; margin: 4px; }
    .stButton>button:hover { background-color: #d0d0d0; }
</style>
""", unsafe_allow_html=True)


# --- Prompts ---
DR_ANITA_SCHOTT_PROMPT = """
You are Dr. Anita Schott, a pediatric infectious disease specialist and immunization expert with over 20 years of clinical experience.
Your identity is Dr. Anita Schott. You will always speak in the first person ("I," "my").
You must never reveal that you are an AI.

## Persona & Tone
* **Identity:** Experienced, practical, and empathetic attending physician.
* **Tone:** Professional, calm, and confident.

## !! MANDATORY: Core Interaction Loop !!
You MUST follow this logic flow for every user question.

1. Analyze the User's Query: First, read the {question}.
2. Triage: Is it a Recommendation or a Factual Question?
    * A) Factual Question: Answer it directly using {context}.
    * B) Recommendation/Advice Question:
        * 1. Check {chat_history}. Do you have patient info (age, location)?
        * 2. If NO: Ask clarifying questions.
        * 3. If YES: Provide recommendation.

Context:
{context}

History:
{chat_history}

Question:
{question}

Answer:
"""

DR_ANITA_SCHOTT_PERSONA_ONLY_PROMPT = """
You are Dr. Anita Schott. Respond to small talk naturally, briefly, and in character.
"""

INTENT_CLASSIFICATION_PROMPT = """
You are an expert intent classifier. Categorize the query into:
1. knowledge_query (medical/factual)
2. conversational_query (small talk/greeting)

User Query: "{user_input}"
Intent:
"""


# --- Core Logic ---

@st.cache_resource
def load_knowledge_base():
    FAISS_INDEX_PATH = "faiss_index"
    if not os.path.exists(FAISS_INDEX_PATH):
        st.sidebar.warning(f"Knowledge base not found at '{FAISS_INDEX_PATH}'. Chatbot will use general knowledge only.")
        return None
    try:
        st.sidebar.info("Loading knowledge base...")
        # Since we set os.environ["OPENAI_API_KEY"], we don't need to pass it explicitly
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        st.sidebar.success("Knowledge base loaded!")
        return vector_store
    except Exception as e:
        st.sidebar.error(f"Failed to load knowledge base: {e}")
        return None

@st.cache_resource
def get_intent_classifier():
    # Since we set os.environ["OPENAI_API_KEY"], we don't need to pass it explicitly
    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    prompt = PromptTemplate(template=INTENT_CLASSIFICATION_PROMPT, input_variables=["user_input"])
    return LLMChain(llm=model, prompt=prompt)

def get_conversational_chain(_vector_store, _memory):
    model = ChatOpenAI(model_name="gpt-4o", temperature=0.3)
    
    qa_prompt = PromptTemplate(
        template=DR_ANITA_SCHOTT_PROMPT, 
        input_variables=["context", "question", "chat_history"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=_vector_store.as_retriever(search_kwargs={"k": 8}),
        memory=_memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=True,
        rephrase_question=False 
    )
    return chain

def handle_conversational_query(query, memory):
    model = ChatOpenAI(model_name="gpt-4o", temperature=0.3)
    messages = [SystemMessage(content=DR_ANITA_SCHOTT_PERSONA_ONLY_PROMPT)]
    chat_history_messages = memory.chat_memory.messages
    messages.extend(chat_history_messages)
    messages.append(HumanMessage(content=query))
    response = model.invoke(messages)
    memory.save_context({"question": query}, {"answer": response.content})
    return response.content

# --- Main Streamlit UI Logic ---
st.title("💬 Chat with Dr. Anita Schott")

# Initialize Session State
if "vector_store" not in st.session_state:
    st.session_state.vector_store = load_knowledge_base()
if "intent_classifier" not in st.session_state:
    st.session_state.intent_classifier = get_intent_classifier()
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True, output_key='answer'
    )
if "chain" not in st.session_state and st.session_state.vector_store:
    st.session_state.chain = get_conversational_chain(
        st.session_state.vector_store, st.session_state.memory
    )
if "messages" not in st.session_state:
    st.session_state.messages = []
    if "chain" in st.session_state and st.session_state.chain:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Hello, I'm Dr. Anita Schott. How can I help you today?"
        })

# Display Chat
for message in st.session_state.messages:
    role_class = "user-bubble" if message["role"] == "user" else "assistant-bubble"
    st.markdown(f'<div class="chat-bubble {role_class}">{message["content"]}</div>', unsafe_allow_html=True)

# Handle Chat
def handle_chat(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})

    if "chain" in st.session_state and st.session_state.chain:
        # Intent Classification
        intent_result = st.session_state.intent_classifier.invoke({"user_input": prompt})
        intent = intent_result['text'].strip().lower()
        
        if "knowledge_query" in intent:
            response = st.session_state.chain({"question": prompt})
            answer = response['answer']
        else:
            answer = handle_conversational_query(prompt, st.session_state.memory)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "The knowledge base is not loaded. Please ensure 'ingest.py' has been run and parts are uploaded."
        })
    st.rerun()

# Suggestions
if len(st.session_state.get("messages", [])) <= 1:
    st.markdown("---")
    cols = st.columns(3)
    prompts = ["Importance of vaccination", "Newborn safety measures", "Vaccination schedule?"]
    if cols[0].button(prompts[0]): handle_chat(prompts[0])
    if cols[1].button(prompts[1]): handle_chat(prompts[1])
    if cols[2].button(prompts[2]): handle_chat(prompts[2])

# Input
if prompt := st.chat_input("Ask Dr. Anita Schott..."):
    handle_chat(prompt)
