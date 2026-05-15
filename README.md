# 🔧 AI Race Engineer — F1 Chatbot powered by Claude API

> **Part 4 of the [Learning AI/ML with F1](https://github.com/anurag-lenkaa) series by Divyanshi Kamra**

Chat with an AI F1 Race Engineer that queries **real historical data** to answer your questions. Built with the **Anthropic Claude API** using **tool use** — Claude decides which data tool to call, fetches the answer, and responds like an expert engineer.

---

## 🏁 What You'll Learn
- **Anthropic Claude API** — messages, system prompts, model selection
- **Tool use (function calling)** — letting Claude call your Python functions
- **Agentic loops** — handling multi-turn tool use automatically
- **Streamlit chat UI** — `st.chat_message`, `st.chat_input`, session state
- Building **data-grounded AI** that doesn't hallucinate

---

## 🛠️ How Tool Use Works

```python
client = anthropic.Anthropic(api_key="your-key")

response = client.messages.create(
    model="claude-sonnet-4-5",
    tools=TOOLS,           # 5 F1 data tools
    messages=history,
)

if response.stop_reason == "tool_use":
    # Claude called a tool — execute it and feed results back
    result = execute_tool(block.name, block.input)
    # Loop continues until Claude gives a final text response
```

---

## 🔨 Available Tools (5)

| Tool | What it does |
|------|-------------|
| `get_driver_stats` | Career wins, championships, nationality, teams |
| `get_champion_by_year` | Driver + constructor champion for any year |
| `get_constructor_stats` | Team wins, seasons active, best results |
| `search_race_results` | Search by driver, circuit, country, year |
| `get_top_drivers` | Top N drivers by race wins (filterable by decade) |

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

# Set your API key
set ANTHROPIC_API_KEY=sk-ant-...   # Windows
export ANTHROPIC_API_KEY=sk-ant-... # Mac/Linux

# Launch the chatbot
streamlit run app.py
```

> Get your free API key at [console.anthropic.com](https://console.anthropic.com)

---

## 🛠️ Tech Stack

![Claude](https://img.shields.io/badge/Claude-claude--sonnet--4--5-blueviolet)
![Anthropic](https://img.shields.io/badge/Anthropic-API-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Chat-FF4B4B?logo=streamlit)
![Pandas](https://img.shields.io/badge/Pandas-2.0-150458?logo=pandas)

- `anthropic` — Claude API with tool use
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

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Set **Main file:** `app.py`
4. Add secret: `ANTHROPIC_API_KEY = "sk-ant-..."`
5. Deploy 🚀

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
