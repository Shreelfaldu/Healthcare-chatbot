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

# --- Define Color Palette ---
PRIMARY_COLOR = "#504B38"      # Dark Brown/Khaki for text, primary elements
SECONDARY_COLOR = "#B9B28A"    # Muted Olive/Khaki for accents, borders
ACCENT_COLOR = "#EBE5C2"       # Pale Yellow for secondary backgrounds, highlights
BACKGROUND_COLOR = "#F8F3D9"   # Light Cream for main background

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Medical Assistant",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Styling ---
def custom_css():
    return f"""
<style>
    /* General Body Styling */
    body {{
        background-color: {BACKGROUND_COLOR};
        color: {PRIMARY_COLOR};
        font-family: 'Arial', sans-serif; /* Example font */
    }}

    /* Streamlit Core Components */
    .stApp {{
        background-color: {BACKGROUND_COLOR};
        color: {PRIMARY_COLOR};
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {ACCENT_COLOR};
        padding: 20px;
        border-radius: 10px;
    }}
    .stSidebarContent {{
        padding: 20px;
    }}
    .stSidebarContent h2, .stSidebarContent h3 {{
        color: {PRIMARY_COLOR};
    }}
    .stSidebarContent .stButton>button {{
        width: 100%;
        background-color: {PRIMARY_COLOR};
        color: {BACKGROUND_COLOR};
        border: none;
        padding: 10px 18px;
        margin-bottom: 10px;
        border-radius: 5px;
        font-weight: bold;
    }}
    .stSidebarContent .stButton>button:hover {{
        background-color: {SECONDARY_COLOR};
        color: {PRIMARY_COLOR};
    }}
    .stSidebarContent .stButton>button[kind="secondary"] {{ /* For delete button if needed */
        background-color: #f0f2f6; /* Lighter default for delete */
        color: {PRIMARY_COLOR};
    }}

    /* Main Chat Area */
    .stChatInputContainer {{
        background-color: {ACCENT_COLOR};
        border-radius: 10px;
        padding: 10px;
        margin-top: 20px;
    }}
    .stChatInputContainer textarea {{
        background-color: {ACCENT_COLOR};
        color: {PRIMARY_COLOR};
        border: 1px solid {SECONDARY_COLOR};
    }}
    .stChatInputContainer textarea::placeholder {{
        color: {PRIMARY_COLOR} !important;
        opacity: 0.7;
    }}
    .stTextInput>div>div>input {{
        background-color: {ACCENT_COLOR};
        color: {PRIMARY_COLOR};
        border: 1px solid {SECONDARY_COLOR};
    }}

    /* Chat Messages */
    .stChatMessage {{
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }}
    .stChatMessage[data-testid="stChatMessage"] {{ /* Targeting the container div for messages */
        background-color: {BACKGROUND_COLOR}; /* Default for messages */
        border: 1px solid {SECONDARY_COLOR};
    }}
    .stChatMessage:not([data-testid="stChatMessage"]):not([data-testid="stChatMessageUser"]) {{ /* Assistant messages */
        background-color: {ACCENT_COLOR};
        border: 1px solid {SECONDARY_COLOR};
        color: {PRIMARY_COLOR};
    }}
    .stChatMessage[data-testid="stChatMessageUser"] {{ /* User messages */
        background-color: {EBE5C2}; /* Using accent for user messages for contrast */
        color: {PRIMARY_COLOR};
        border: 1px solid {SECONDARY_COLOR};
    }}
    .stChatMessage .message-content {{ /* If you can target the content div directly */
        color: {PRIMARY_COLOR};
    }}

    /* Title and Headers */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
        color: {PRIMARY_COLOR};
    }}

    /* Buttons */
    .stButton>button {{
        background-color: {PRIMARY_COLOR};
        color: {BACKGROUND_COLOR};
        border: none;
        padding: 10px 20px;
        margin: 5px 0;
        border-radius: 5px;
        font-weight: bold;
    }}
    .stButton>button:hover {{
        background-color: {SECONDARY_COLOR};
        color: {PRIMARY_COLOR};
    }}
    .stButton>button[kind="secondary"] {{ /* Styling for 'New Chat' button if it were secondary */
        background-color: {SECONDARY_COLOR};
        color: {PRIMARY_COLOR};
    }}

    /* Loading Spinners */
    .stSpinner {{
        color: {PRIMARY_COLOR};
    }}

    /* Custom Scrollbar */
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    ::-webkit-scrollbar-track {{
        background: {ACCENT_COLOR};
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb {{
        background: {SECONDARY_COLOR};
        border-radius: 4px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: {PRIMARY_COLOR};
    }}
</style>
"""

st.markdown(custom_css(), unsafe_allow_html=True)

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

# Prompt template for main assistant
chat_prompt = ChatPromptTemplate.from_template("""
You are a highly reliable and knowledgeable AI medical assistant. Your job is to provide accurate, evidence-based information about medical topics such as symptoms, diseases, treatments, preventions, medications, and health tips.

Always respond in a clear and concise manner, using plain language understandable to a general audience.

If a question seems to require a personal diagnosis, medical opinion, or is an emergency, respond with:
"I'm not qualified to provide that information. Please consult a licensed healthcare professional."

Here is the previous conversation for context:
{chat_history}

Respond to the following user message in a {tone} tone:
"{topic}"
""")

# Prompt template for title generation
title_prompt = ChatPromptTemplate.from_template("""
Give a short, 3 to 6 word topic title for this medical query:
"{query}"
""")

# ---------------------------
# Initialize Session State
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None # Explicitly set to None initially
if "title_set" not in st.session_state:
    st.session_state.title_set = False # Explicitly set to False initially

# --- Helper Function to Get or Create Chat ID ---
def get_or_create_chat_id():
    if st.session_state.chat_id is None:
        # Create a new chat ID if none exists
        st.session_state.chat_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return st.session_state.chat_id

# ---------------------------
# Sidebar - Chat History
# ---------------------------
with st.sidebar:
    st.subheader("⚙️ Settings", divider='rainbow')
    tone = st.selectbox("Choose tone:", ["professional", "friendly", "reassuring", "neutral"], key="tone")
    st.markdown("---")
    st.title("📜 Chats")

    chat_dirs = sorted(os.listdir("chat_history"))
    chat_titles = []
    for folder in chat_dirs:
        meta_path = f"chat_history/{folder}/metadata.json"
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    title = json.load(f).get("title", folder)
            except (FileNotFoundError, json.JSONDecodeError):
                title = folder # Fallback if metadata is bad
        else:
            title = folder
        chat_titles.append((title, folder))

    for i, (title, folder) in enumerate(chat_titles):
        cols = st.columns([0.75, 0.25])
        with cols[0]:
            if st.button(title, key=f"load_{i}"):
                try:
                    chat_path = f"chat_history/{folder}/chat.json"
                    with open(chat_path, "r") as f:
                        st.session_state.messages = json.load(f)
                    st.session_state.chat_id = folder
                    st.session_state.title_set = True
                    st.rerun()
                except FileNotFoundError:
                    st.error(f"Chat file not found for {title}.")
                except json.JSONDecodeError:
                    st.error(f"Error decoding chat history for {title}.")

        with cols[1]:
            if st.button("🗑️", key=f"delete_{i}", help=f"Delete chat: {title}"):
                if st.session_state.get("chat_id") == folder:
                    st.session_state.messages = []
                    st.session_state.chat_id = None
                    st.session_state.title_set = False
                import shutil
                shutil.rmtree(f"chat_history/{folder}", ignore_errors=True)
                st.success(f"Deleted chat: {title}")
                st.rerun()

    # Styling for the "New Chat" button
    if st.button("🆕 New Chat", key="new_chat_button"):
        st.session_state.messages = []
        st.session_state.chat_id = None # Reset chat_id to trigger creation in get_or_create_chat_id
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

    # Get or create chat_id before using it
    current_chat_id = get_or_create_chat_id()

    # Set chat title from first user message using LLM
    if not st.session_state.title_set:
        meta_path = f"chat_history/{current_chat_id}/metadata.json"
        os.makedirs(os.path.dirname(meta_path), exist_ok=True) # Ensure directory exists

        try:
            title_chain = LLMChain(llm=llm, prompt=title_prompt)
            title_result = title_chain.invoke({"query": user_input})
            brief_title = title_result["text"].strip()
        except Exception as e:
            brief_title = user_input.strip().split()[0:5]  # fallback
            brief_title = " ".join(brief_title)

        with open(meta_path, "w") as f:
            json.dump({"title": brief_title}, f)
        st.session_state.title_set = True
        # Rerun here. If chat_id was just created, this rerun will ensure
        # the sidebar is updated, and the main area continues with the new ID.
        st.rerun()


    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare chat history for context (exclude current user message)
                history_text = ""
                for msg in st.session_state.messages[:-1]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_text += f"{role}: {msg['content']}\n"

                # Invoke LLM with context
                chain = LLMChain(llm=llm, prompt=chat_prompt)
                result = chain.invoke({
                    "topic": user_input,
                    "tone": st.session_state.tone,
                    "chat_history": history_text
                })
                reply = result["text"].strip()
            except Exception as e:
                reply = f"⚠️ Error: {str(e)}"

            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

    # Auto-save chat final version
    # Ensure chat_id is set before attempting to save
    if current_chat_id: # Use the guaranteed chat_id
        chat_path = f"chat_history/{current_chat_id}/chat.json"
        os.makedirs(os.path.dirname(chat_path), exist_ok=True)
        with open(chat_path, "w") as f:
            json.dump(st.session_state.messages, f, indent=2)
