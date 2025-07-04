# 🩺 AI Medical Assistant Chatbot

A powerful, interactive AI-powered medical assistant built with **LangChain**, **Streamlit**, and **Azure OpenAI (GPT-4o)**.  
It provides concise, evidence-based answers to general health-related queries in a variety of tones for a better user experience.

---

## 🚀 Features

- ✅ **Powered by GPT-4o (Azure OpenAI)**
- ✅ **Streamlit interface** for real-time Q&A
- ✅ **Customizable tone**: Professional, Friendly, Reassuring, or Neutral
- ✅ **Auto-generated titles** for each conversation
- ✅ **Local chat history saving, loading & deletion**
- ✅ **Safe fallback responses** for personal diagnosis or emergencies
- ✅ **Secure .env-based Azure credentials handling**
- ✅ **Fully custom UI with modern responsive styling (ColorHunt palette)**

---

## 🧠 How It Works

The app leverages prompt engineering and LangChain’s LLM orchestration to simulate a virtual healthcare assistant:

> “You are a highly reliable and knowledgeable AI medical assistant...  
> If a question requires diagnosis, respond:  
> *I'm not qualified to provide that information. Please consult a licensed healthcare professional.*”

💬 Each user query is interpreted by GPT-4o with previous chat history as context, a selected tone of voice, and a safety-first prompt structure.

---

## 🔗 Live Preview

🚀 Try the live app here: [AI Medical Assistant Chatbot](https://healthcares.streamlit.app/)

---

## 🛠️ Tech Stack

| Tool              | Purpose                         |
|------------------|---------------------------------|
| **Streamlit**     | Frontend UI for the chatbot     |
| **Azure OpenAI**  | LLM backend (GPT-4o)            |
| **LangChain**     | Prompt templates & LLM chaining |
| **python-dotenv** | Secure API key handling         |
| **Custom CSS**    | Enhanced UI design              |

---

## 🎨 UI Design Theme

This app uses a customized healthcare-friendly UI based on a palette from [ColorHunt](https://colorhunt.co/palette/f8f3d9ebe5c2b9b28a504b38):


All components, from the sidebar to the chat interface and buttons, are restyled using advanced CSS for a soft, professional look.

---

## 🧪 Setup & Run Locally

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/healthcare-assistant-bot.git
cd healthcare-assistant-bot
```
```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```
```bash
pip install -r requirements.txt
```

---
Create a .env file in the root directory:
```bash
streamlit run app.py
```

📁 Folder Structure
```bash
.
├── app.py                  # Main Streamlit app
├── chat_history/           # Saved chat sessions
├── .env                    # Environment secrets
├── requirements.txt        # Required dependencies
└── README.md               # Project documentation
```
⚠️ Disclaimer
This AI assistant provides general medical information for educational purposes only.
It is not a substitute for professional medical advice, diagnosis, or treatment.
Always consult a licensed healthcare provider regarding your health.

👨‍💻 Author
Shreel Faldu
