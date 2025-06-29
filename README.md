# 🩺 AI Medical Assistant Chatbot

A powerful, interactive AI-powered medical assistant built with **LangChain**, **Streamlit**, and **Azure OpenAI (GPT-4o)**.  
It provides concise, evidence-based answers to general health-related queries in a variety of tones for a better user experience.

---

## 🚀 Features

- ✅ **Powered by GPT-4o (Azure OpenAI)**
- ✅ **Streamlit interface** for real-time question & answer
- ✅ **Customizable tone**: Professional, Friendly, Reassuring, or Neutral
- ✅ **Chat history**: Scrollable assistant and user messages
- ✅ **Safe fallback responses** for diagnosis or emergencies
- ✅ **Supports general health topics** including diseases, symptoms, treatments, preventions, and wellness

---

## 🧠 How It Works

The app uses a structured prompt to guide the LLM with a focus on medical safety and simplicity:

> “You are a highly reliable and knowledgeable AI medical assistant... If a question requires diagnosis, respond:  
> *I'm not qualified to provide that information. Please consult a licensed healthcare professional.*”

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **LangChain** | Prompting and chaining |
| **Azure OpenAI** | LLM backend (GPT-4o) |
| **Streamlit** | Web interface |
| **Python-dotenv** | Secure environment management |

---

## 🧪 Setup & Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/healthcare-assistant-bot.git
cd healthcare-assistant-bot
