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

# Prompt templates
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

title_prompt = ChatPromptTemplate.from_template("""
Give a short, 3 to 6 word topic title for this medical query:
"{query}"
""")

# ---------------------------
# Enhanced Custom CSS Styling
# ---------------------------
def load_css():
    st.markdown("""
    <style>
    /* Color Variables */
    :root {
        --primary-dark: #504B38;
        --primary-medium: #B9B28A;
        --primary-light: #EBE5C2;
        --primary-lightest: #F8F3D9;
        --text-dark: #2c2c2c;
        --text-light: #ffffff;
        --shadow: 0 4px 6px rgba(0,0,0,0.1);
        --shadow-hover: 0 8px 15px rgba(0,0,0,0.2);
        --border-radius: 15px;
        --border-radius-small: 8px;
        --animation-fast: 0.2s;
        --animation-slow: 0.4s;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main app container */
    .main .block-container {
        padding: 2rem 1rem;
        background: linear-gradient(135deg, var(--primary-lightest) 0%, var(--primary-light) 100%);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc, .css-17eq0hr {
        background: linear-gradient(180deg, var(--primary-dark) 0%, var(--primary-medium) 100%);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
    }
    
    /* Sidebar content */
    .css-1d391kg *, .css-1lcbmhc *, .css-17eq0hr * {
        color: var(--text-light) !important;
    }
    
    /* Title styling */
    .main-title {
        text-align: center;
        color: var(--text-light);
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        padding: 2rem;
        # background: linear-gradient(135deg, var(--primary-lightest) 0%, var(--primary-medium) 50%, var(--primary-lightest) 100%);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        animation: titleGlow 2s ease-in-out infinite alternate;
    }
    
    @keyframes titleGlow {
        from { box-shadow: var(--shadow); }
        to { box-shadow: var(--shadow-hover); }
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: var(--text-lightest);
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-style: italic;
        font-weight: 500;
        opacity: 0.8;
    }
    
    /* Chat message container improvements */
    [data-testid="stChatMessage"] {
        background: var(--primary-lightest);
        border-radius: var(--border-radius);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: var(--shadow);
        transition: all var(--animation-fast) ease;
        border: 1px solid var(--primary-light);
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="stChatMessage"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--primary-medium), var(--primary-dark));
        opacity: 0;
        transition: opacity var(--animation-fast) ease;
    }
    
    [data-testid="stChatMessage"]:hover::before {
        opacity: 1;
    }
    
    [data-testid="stChatMessage"]:hover {
        box-shadow: var(--shadow-hover);
        transform: translateY(-2px);
    }
    
    /* User message styling */
    [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] {
        background: var(--primary-dark);
        color: var(--text-light);
        border-radius: 50%;
        padding: 0.5rem;
    }
    
    /* Assistant message styling */
    [data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
        background: var(--primary-medium);
        color: var(--text-light);
        border-radius: 50%;
        padding: 0.5rem;
    }
    
    /* Chat message content */
    [data-testid="stChatMessage"] .stMarkdown {
        color: var(--text-dark);
        line-height: 1.6;
    }
    
    [data-testid="stChatMessage"] .stMarkdown p {
        font-size: 1rem;
    }
    
    /* Enhanced chat input styling */
    .stChatFloatingInputContainer {
        background: var(--primary-light);
        border: 2px solid var(--primary-dark);
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        padding: 0.8rem;
        margin: 1rem 0;
        transition: all var(--animation-fast) ease;
    }
    
    .stChatFloatingInputContainer:hover {
        box-shadow: var(--shadow-hover);
        transform: translateY(-1px);
    }
    
    .stChatFloatingInputContainer input {
        background: var(--primary-lightest);
        border: 1px solid var(--primary-medium);
        border-radius: var(--border-radius-small);
        color: var(--text-dark);
        padding: 0.75rem;
        font-size: 1rem;
        transition: all var(--animation-fast) ease;
    }
    
    .stChatFloatingInputContainer input:focus {
        border-color: var(--primary-dark);
        outline: none;
    }
    
    .stChatFloatingInputContainer button {
        background: var(--primary-dark);
        border: none;
        border-radius: var(--border-radius-small);
        color: var(--text-light);
        padding: 0.75rem;
        transition: all var(--animation-fast) ease;
    }
    
    .stChatFloatingInputContainer button:hover {
        background: var(--primary-medium);
        transform: scale(1.05);
    }
    
    /* Enhanced button styling */
    .stButton > button {
        # background: linear-gradient(135deg, var(--primary-medium), var(--primary-dark));
        background: transparent;
        color: var(--text-light);
        border: 1px solid var(--primary-lightest);
        border-radius: var(--border-radius-small);
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all var(--animation-fast) ease;
        box-shadow: var(--shadow);
        cursor: pointer;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-hover);
        background: var(--primary-medium);
        color: var(--text-dark);
    }
    
    /* New chat button special styling */
    .stButton > button:contains("🆕") {
        background: var(--primary-medium);
        font-weight: 700;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .stButton > button:contains("🆕"):hover {
        background: var(--primary-medium);
    }
    
    /* Delete button styling */
    .stButton > button:contains("🗑️") {
        background: var(--primary-medium);
        padding: 0.5rem;
        font-size: 0.9rem;
        min-width: 40px;
    }
    
    .stButton > button:contains("🗑️"):hover {
        background: var(--primary-medium);
    }
    
    /* Enhanced selectbox styling */
    .stSelectbox > div > div {
        background: var(--primary-light);
        border: 2px solid var(--primary-medium);
        border-radius: var(--border-radius-small);
        color: var(--text-dark);
        transition: all var(--animation-fast) ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--primary-dark);
        box-shadow: 0 0 0 2px rgba(80, 75, 56, 0.1);
    }
    
    /* Spinner styling */
    .stSpinner {
        color: var(--primary-dark);
    }
    
    /* Success message styling */
    .stSuccess {
        background: var(--primary-medium);
        color: var(--text-light);
        border-radius: var(--border-radius-small);
        padding: 1rem;
        box-shadow: var(--shadow);
        animation: slideIn var(--animation-slow) ease;
    }
    
    /* Info message styling */
    .stInfo {
        background: var(--primary-medium);
        color: var(--text-dark);
        border-radius: var(--border-radius-small);
        padding: 1rem;
        box-shadow: var(--shadow);
        border-left: 4px solid var(--primary-dark);
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(to right, transparent, var(--primary-medium), transparent);
        margin: 1.5rem 0;
        opacity: 0.6;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: var(--primary-medium);
        font-size: 0.9rem;
        margin-top: 2rem;
        padding: 1rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: var(--border-radius-small);
        backdrop-filter: blur(10px);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.2rem;
            padding: 1rem;
        }
        
        .subtitle {
            font-size: 1rem;
        }
        
        [data-testid="stChatMessage"] {
            padding: 1rem;
            margin: 0.5rem 0;
        }
        
        .stChatFloatingInputContainer {
            padding: 0.5rem;
        }
    }
    
    /* Animation for new messages */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    [data-testid="stChatMessage"] {
        animation: slideIn var(--animation-slow) ease-out;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--primary-light);
        border-radius: var(--border-radius-small);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-medium);
        border-radius: var(--border-radius-small);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary-dark);
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------------------
# Initialize Session State
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_id" not in st.session_state:
    st.session_state.chat_id = None
if "title_set" not in st.session_state:
    st.session_state.title_set = False

# Load custom CSS
load_css()

# ---------------------------
# Enhanced Sidebar
# ---------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    # Tone selection
    tone = st.selectbox(
        "Choose your preferred tone:",
        ["professional", "friendly", "reassuring", "neutral"],
        key="tone",
        help="Select how you'd like the assistant to respond"
    )
    
    st.markdown("---")
    
    # Chat history section
    st.markdown("### 📜 Chat History")
    
    # New chat button
    if st.button("🆕 Start New Chat", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        st.session_state.title_set = False
        st.rerun()
    
    st.markdown("---")
    
    # Chat history list
    if os.path.exists("chat_history"):
        chat_dirs = sorted(os.listdir("chat_history"), reverse=True)
        chat_titles = []
        
        for folder in chat_dirs:
            meta_path = f"chat_history/{folder}/metadata.json"
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        title = json.load(f).get("title", folder)
                except:
                    title = folder
            else:
                title = folder
            chat_titles.append((title, folder))
        
        if chat_titles:
            st.markdown("**Recent Conversations:**")
            for i, (title, folder) in enumerate(chat_titles[:10]):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    display_title = title if len(title) <= 25 else title[:22] + "..."
                    if st.button(
                        f"💬 {display_title}",
                        key=f"load_{i}",
                        help=f"Load chat: {title}",
                        use_container_width=True
                    ):
                        try:
                            with open(f"chat_history/{folder}/chat.json", "r") as f:
                                st.session_state.messages = json.load(f)
                            st.session_state.chat_id = folder
                            st.session_state.title_set = True
                            st.rerun()
                        except:
                            st.error("Failed to load chat history")
                
                with col2:
                    if st.button("🗑️", key=f"delete_{i}", help="Delete this chat"):
                        if st.session_state.get("chat_id") == folder:
                            st.session_state.messages = []
                            st.session_state.chat_id = None
                            st.session_state.title_set = False
                        
                        import shutil
                        shutil.rmtree(f"chat_history/{folder}", ignore_errors=True)
                        st.success(f"✅ Deleted: {title}")
                        st.rerun()
        else:
            st.info("No previous conversations found. Start a new chat to begin!")

# ---------------------------
# Main Chat Interface
# ---------------------------
# Enhanced title
st.markdown('<h1 class="main-title">🩺 AI Medical Assistant</h1>', unsafe_allow_html=True)

# Subtitle
st.markdown("""
<div class="subtitle">
    Get reliable medical information and health guidance. Always consult healthcare professionals for personal medical advice.
</div>
""", unsafe_allow_html=True)

# Chat messages display
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Enhanced chat input
if user_input := st.chat_input("💬 Ask your medical question here... (e.g., 'What are the symptoms of flu?')"):
    # Add user message
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate chat title from first message
    if not st.session_state.title_set:
        if not st.session_state.chat_id:
            st.session_state.chat_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        meta_path = f"chat_history/{st.session_state.chat_id}/metadata.json"
        os.makedirs(f"chat_history/{st.session_state.chat_id}", exist_ok=True)

        try:
            title_chain = LLMChain(llm=llm, prompt=title_prompt)
            title_result = title_chain.invoke({"query": user_input})
            brief_title = title_result["text"].strip()
        except Exception as e:
            # Fallback title generation
            words = user_input.strip().split()[:5]
            brief_title = " ".join(words)
            if len(brief_title) > 30:
                brief_title = brief_title[:27] + "..."

        with open(meta_path, "w") as f:
            json.dump({"title": brief_title, "created": datetime.datetime.now().isoformat()}, f)
        st.session_state.title_set = True

    # Get assistant response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Analyzing your question..."):
            try:
                # Prepare chat history context
                history_text = ""
                for msg in st.session_state.messages[:-1]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_text += f"{role}: {msg['content']}\n"

                # Get LLM response
                chain = LLMChain(llm=llm, prompt=chat_prompt)
                result = chain.invoke({
                    "topic": user_input,
                    "tone": st.session_state.tone,
                    "chat_history": history_text
                })
                reply = result["text"].strip()
                
            except Exception as e:
                reply = f"⚠️ I'm experiencing technical difficulties. Please try again later.\n\nError details: {str(e)}"

            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})

    # Auto-save chat
    if st.session_state.chat_id:
        chat_path = f"chat_history/{st.session_state.chat_id}/chat.json"
        with open(chat_path, "w") as f:
            json.dump(st.session_state.messages, f, indent=2)

# ---------------------------
# Enhanced Footer
# ---------------------------
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>⚠️ <strong>Disclaimer:</strong> This AI assistant provides general medical information for educational purposes only. 
    Always consult qualified healthcare professionals for medical advice, diagnosis, or treatment.</p>
    <p style="margin-top: 1rem;">🔒 Your conversations are saved locally and remain private.</p>
</div>
""", unsafe_allow_html=True)
