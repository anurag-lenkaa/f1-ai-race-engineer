# 🔧 AI Race Engineer — F1 Chatbot powered by Gemini AI

> **Part 4 of the [Learning AI/ML with F1](https://github.com/anurag-lenkaa) series by Divyanshi Kamra**

Chat with an AI F1 Race Engineer that queries **real historical data** to answer your questions. Built with the **Google Gemini API** (free!) using **function calling** — Gemini decides which data tool to call, fetches the answer, and responds like an expert engineer.

---

## 🏁 What You'll Learn
- **Google Gemini API** — messages, system prompts, model selection
- **Function calling** — letting Gemini call your Python functions automatically
- **Agentic loops** — handling multi-turn tool use automatically
- **Streamlit chat UI** — `st.chat_message`, `st.chat_input`, session state
- Building **data-grounded AI** that doesn't hallucinate

---

## 🛠️ How Function Calling Works

```python
import google.generativeai as genai

genai.configure(api_key="your-key")

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",   # Free tier
    tools=[get_driver_stats, get_champion_by_year, ...],  # Your Python functions
    system_instruction="You are an expert F1 Race Engineer..."
)

chat = model.start_chat(enable_automatic_function_calling=True)
response = chat.send_message("Who has the most F1 wins?")
# Gemini automatically calls your functions and returns final answer
print(response.text)
```

---

## 🔨 Available Tools (5)

| Tool | What it does |
|------|-------------|
| `get_driver_stats` | Career wins, championships, nationality, teams |
| `get_champion_by_year` | Driver + constructor champion for any year |
| `get_constructor_stats` | Team wins, seasons active, best results |
| `search_race_results` | Search by driver, circuit, country, year |
| `get_top_drivers` | Top N drivers by race wins |

---

## 💬 Example Questions

- *"Who has the most F1 wins of all time?"*
- *"Tell me about Lewis Hamilton's career stats"*
- *"Who won the 2021 championship and with which team?"*
- *"Which constructor dominated the 2010s?"*
- *"Compare Hamilton and Schumacher"*

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt

# Set your free Gemini API key
# Get it free at: https://aistudio.google.com/apikey

set GOOGLE_API_KEY=your-key-here     # Windows
export GOOGLE_API_KEY=your-key-here  # Mac/Linux

# Launch the chatbot
streamlit run app.py
```

> Get your **free** API key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — no credit card needed!

---

## ☁️ Deploy on Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set **Repository:** `anurag-lenkaa/f1-ai-race-engineer`
4. Set **Main file:** `app.py`
5. Click **Advanced settings** → **Secrets** → add:
   ```
   GOOGLE_API_KEY = "your-key-here"
   ```
6. Deploy 🚀

---

## 🛠️ Tech Stack

![Gemini](https://img.shields.io/badge/Gemini-1.5%20Flash-blue?logo=google)
![Google](https://img.shields.io/badge/Google-AI%20Studio-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Chat-FF4B4B?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas)

- `google-generativeai` — Gemini API with function calling (free tier)
- `pandas` — data querying for tool functions
- `streamlit` — chat interface with history

---

## 📁 File Structure
```
ai-race-engineer/
├── ai_race_engineer_chatbot.ipynb  ← Full notebook with explanations
├── app.py                          ← Streamlit chat app
├── data/
│   ├── race_results.csv
│   ├── driver_champions.csv
│   └── constructor_champions.csv
├── requirements.txt
└── README.md
```

---

## 🔗 Part of the Series
| Part | Project | Repo |
|------|---------|------|
| 1 | F1 Data Analysis | [f1-data-analysis](https://github.com/anurag-lenkaa/f1-data-analysis) |
| 2 | F1 Race Telemetry | [f1-race-telemetry](https://github.com/anurag-lenkaa/f1-race-telemetry) |
| 3 | F1 Race Predictor | [f1-race-predictor](https://github.com/anurag-lenkaa/f1-race-predictor) |
| 4 | **AI Race Engineer** | You are here |

---

> Series by [Divyanshi Kamra](https://www.instagram.com/divyanshi.kamra) | Built by [anurag-lenkaa](https://github.com/anurag-lenkaa)
