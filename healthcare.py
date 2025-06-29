import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Set up Azure LLM
llm = AzureChatOpenAI(
    openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    model_name="gpt-4o",
    temperature=0.5,
)

# Chat prompt with contextual instructions
prompt = ChatPromptTemplate.from_template("""
You are a highly reliable and knowledgeable AI medical assistant. Your job is to provide accurate, evidence-based information about medical topics such as symptoms, diseases, treatments, preventions, medications, and health tips.

Always respond in a clear and concise manner, using plain language understandable to a general audience.

If a question seems to require a personal diagnosis, medical opinion, or is an emergency, respond with:
"I'm not qualified to provide that information. Please consult a licensed healthcare professional."

Respond to the following in a {tone} tone:
"{topic}"
""")

# Streamlit UI
st.set_page_config(page_title="🩺 Advanced Medical Assistant", layout="centered")
st.title("🩺 AI Medical Assistant")

# Sidebar tone and controls
with st.sidebar:
    st.subheader("🔧 Settings")
    tone = st.selectbox("Choose response tone:", ["professional", "friendly", "reassuring", "neutral"])
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

# Input
if user_input := st.chat_input("Ask your medical question..."):
    st.chat_message("user").write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = chain.invoke({"topic": user_input, "tone": tone})
                reply = result["text"].strip()
            except Exception as e:
                reply = f"⚠️ An error occurred: {str(e)}"

            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
