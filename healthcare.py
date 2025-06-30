import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
import os
import datetime
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup directories
os.makedirs("chat_history", exist_ok=True)

# Azure LLM
llm = AzureChatOpenAI(
    openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model_name="gpt-4o",
    temperature=0.5,
)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are a highly reliable and knowledgeable AI medical assistant. Your job is to provide accurate, evidence-based information about medical topics such as symptoms, diseases, treatments, preventions, medications, and health tips.

Always respond in a clear and concise manner, using plain language understandable to a general audience.

If a question seems to require a personal diagnosis, medical opinion, or is an emergency, respond with:
"I'm not qualified to provide that information. Please consult a licensed healthcare professional."

Respond to the following in a {tone} tone:
"{topic}"
""")

# ---------------------------
# Initialize Session State
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None
if "title_set" not in st.session_state:
    st.session_state.title_set = False

# ---------------------------
# Sidebar - Chat History
# ---------------------------

with st.sidebar:
    st.subheader("⚙️ Settings")
    tone = st.selectbox("Choose tone:", ["professional", "friendly", "reassuring", "neutral"], key="tone")
    st.markdown("---")
    st.title("📜 Chats")

    chat_dirs = sorted(os.listdir("chat_history"))
    chat_titles = []
    for folder in chat_dirs:
        meta_path = f"chat_history/{folder}/metadata.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                title = json.load(f).get("title", folder)
        else:
            title = folder
        chat_titles.append((title, folder))

    for i, (title, folder) in enumerate(chat_titles):
        cols = st.columns([0.75, 0.25])
        with cols[0]:
            if st.button(title, key=f"load_{i}"):
                with open(f"chat_history/{folder}/chat.json", "r") as f:
                    st.session_state.messages = json.load(f)
                st.session_state.chat_id = folder
                st.session_state.title_set = True
                st.rerun()
        with cols[1]:
            if st.button("🗑️", key=f"delete_{i}"):
                if st.session_state.get("chat_id") == folder:
                    st.session_state.messages = []
                    st.session_state.chat_id = None
                    st.session_state.title_set = False
                import shutil
                shutil.rmtree(f"chat_history/{folder}", ignore_errors=True)
                st.success(f"Deleted chat: {title}")
                st.rerun()

    if st.button("🆕 New Chat"):
        st.session_state.messages = []
        st.session_state.chat_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        st.session_state.title_set = False
        st.rerun()


# ---------------------------
# Main Chat Area
# ---------------------------
st.title("🩺 AI Medical Assistant")

# Show chat messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Input box
if user_input := st.chat_input("Ask your medical question..."):
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Set chat title from first user message
    if not st.session_state.title_set:
        meta_path = f"chat_history/{st.session_state.chat_id}/metadata.json"
        os.makedirs(f"chat_history/{st.session_state.chat_id}", exist_ok=True)
        with open(meta_path, "w") as f:
            json.dump({"title": user_input.strip()}, f)
        st.session_state.title_set = True

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                chain = LLMChain(llm=llm, prompt=prompt)
                result = chain.invoke({"topic": user_input, "tone": st.session_state.tone})
                reply = result["text"].strip()
            except Exception as e:
                reply = f"⚠️ Error: {str(e)}"

            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

    # Auto-save chat
    chat_path = f"chat_history/{st.session_state.chat_id}/chat.json"
    with open(chat_path, "w") as f:
        json.dump(st.session_state.messages, f, indent=2)
