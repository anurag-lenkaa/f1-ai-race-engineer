import streamlit as st
from google import genai
from google.genai import types
import pandas as pd
import json
import os

st.set_page_config(page_title="AI Race Engineer", page_icon="🔧", layout="wide")
st.title("🔧 AI Race Engineer")
st.markdown(
    "Chat with an F1 expert powered by **Gemini AI** (free!) — queries real race data to answer your questions."
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# ── Sidebar ────────────────────────────────────────────────────────────────────
# Auto-load from Streamlit secrets (for cloud deployment)
_secret_key = st.secrets.get("GOOGLE_API_KEY", "") if hasattr(st, "secrets") else ""

api_key = st.sidebar.text_input(
    "Google Gemini API Key",
    value=_secret_key,
    type="password",
    help="Get yours free at aistudio.google.com/apikey",
)
if _secret_key:
    st.sidebar.success("API key loaded from secrets ✅")
else:
    st.sidebar.markdown("[🔑 Get free API key here](https://aistudio.google.com/apikey)")
st.sidebar.markdown("---")
st.sidebar.markdown("**Example questions:**")
st.sidebar.markdown("- Who has the most F1 wins?")
st.sidebar.markdown("- Tell me about Max Verstappen")
st.sidebar.markdown("- Who won the 2021 championship?")
st.sidebar.markdown("- Which team dominated the 2010s?")
st.sidebar.markdown("- Compare Hamilton and Schumacher")

if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.session_state.history = []
    st.rerun()

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    race_results = pd.read_csv(os.path.join(DATA_DIR, "race_results.csv"))
    driver_champions = pd.read_csv(os.path.join(DATA_DIR, "driver_champions.csv"))
    constructor_champions = pd.read_csv(
        os.path.join(DATA_DIR, "constructor_champions.csv")
    )
    return race_results, driver_champions, constructor_champions


try:
    race_results, driver_champions, constructor_champions = load_data()
    data_loaded = True
except Exception as e:
    st.warning(f"Could not load data: {e}")
    race_results = pd.DataFrame()
    driver_champions = pd.DataFrame()
    constructor_champions = pd.DataFrame()
    data_loaded = False


# ── Helper ─────────────────────────────────────────────────────────────────────
def _find_col(df, *hints):
    for hint in hints:
        for col in df.columns:
            if hint in col.lower():
                return col
    return None


# ── Tool functions — Gemini reads docstrings + type hints for function calling ──
def get_driver_stats(driver_name: str) -> str:
    """Get career statistics for an F1 driver: wins, championships, nationality, teams.

    Args:
        driver_name: Full or partial driver name, e.g. 'Hamilton' or 'Lewis Hamilton'
    """
    if race_results.empty:
        return json.dumps({"error": "Data not available"})

    dcol = _find_col(race_results, "driver")
    if not dcol:
        return json.dumps({"error": "Driver column not found"})

    sub = race_results[race_results[dcol].str.contains(driver_name, case=False, na=False)]
    if sub.empty:
        return json.dumps({"message": f"No results found for '{driver_name}'"})

    result = {"driver": driver_name, "race_entries": len(sub)}

    pos_col = _find_col(sub, "position", "win")
    if pos_col:
        result["wins"] = int((pd.to_numeric(sub[pos_col], errors="coerce") == 1).sum())

    if not driver_champions.empty:
        dchamp_col = _find_col(driver_champions, "driver")
        if dchamp_col:
            champs = driver_champions[
                driver_champions[dchamp_col].str.contains(driver_name, case=False, na=False)
            ]
            result["championships"] = len(champs)
            yr_col = _find_col(champs, "year", "season")
            if yr_col and len(champs):
                result["championship_years"] = sorted(champs[yr_col].tolist())

    ccol = _find_col(sub, "constructor", "team")
    if ccol:
        result["teams"] = sub[ccol].dropna().unique().tolist()

    nat_col = _find_col(sub, "nation", "country")
    if nat_col and len(sub):
        result["nationality"] = sub[nat_col].mode().iloc[0]

    return json.dumps(result, default=str)


def get_champion_by_year(year: int) -> str:
    """Get the F1 World Drivers Champion and Constructors Champion for a given season year.

    Args:
        year: Season year, e.g. 2021
    """
    result = {"year": year}

    if not driver_champions.empty:
        yr_col = _find_col(driver_champions, "year", "season")
        if yr_col:
            row = driver_champions[driver_champions[yr_col] == year]
            if not row.empty:
                dcol = _find_col(driver_champions, "driver")
                result["drivers_champion"] = row[dcol].iloc[0] if dcol else row.iloc[0].to_dict()

    if not constructor_champions.empty:
        yr_col = _find_col(constructor_champions, "year", "season")
        if yr_col:
            row = constructor_champions[constructor_champions[yr_col] == year]
            if not row.empty:
                ccol = _find_col(constructor_champions, "constructor", "team")
                result["constructors_champion"] = (
                    row[ccol].iloc[0] if ccol else row.iloc[0].to_dict()
                )

    if len(result) == 1:
        result["message"] = f"No championship data found for {year}"

    return json.dumps(result, default=str)


def get_constructor_stats(constructor_name: str) -> str:
    """Get career statistics for an F1 constructor/team: total wins and seasons active.

    Args:
        constructor_name: Full or partial team name, e.g. 'Ferrari' or 'Red Bull'
    """
    if race_results.empty:
        return json.dumps({"error": "Data not available"})

    ccol = _find_col(race_results, "constructor", "team")
    if not ccol:
        return json.dumps({"error": "Constructor column not found"})

    sub = race_results[
        race_results[ccol].str.contains(constructor_name, case=False, na=False)
    ]
    if sub.empty:
        return json.dumps({"message": f"No results for '{constructor_name}'"})

    result = {"constructor": constructor_name, "race_entries": len(sub)}

    pos_col = _find_col(sub, "position", "win")
    if pos_col:
        result["wins"] = int((pd.to_numeric(sub[pos_col], errors="coerce") == 1).sum())

    yr_col = _find_col(sub, "year", "season")
    if yr_col:
        years = pd.to_numeric(sub[yr_col], errors="coerce").dropna()
        result["first_season"] = int(years.min())
        result["last_season"] = int(years.max())

    return json.dumps(result, default=str)


def search_race_results(query: str) -> str:
    """Search F1 race results by driver name, circuit, country, or team name.

    Args:
        query: Search term to match against race, driver, constructor, or country fields
    """
    if race_results.empty:
        return json.dumps({"error": "Data not available"})

    str_cols = race_results.select_dtypes(include="object").columns
    mask = race_results[str_cols].apply(
        lambda col: col.str.contains(query, case=False, na=False)
    ).any(axis=1)

    matches = race_results[mask].head(20)
    if matches.empty:
        return json.dumps({"message": f"No results matching '{query}'"})

    return json.dumps(
        {"query": query, "count": int(mask.sum()), "results": matches.to_dict(orient="records")},
        default=str,
    )


def get_top_drivers(top_n: int = 10) -> str:
    """Get the top F1 drivers ranked by total career race wins.

    Args:
        top_n: Number of top drivers to return (default 10)
    """
    if race_results.empty:
        return json.dumps({"error": "Data not available"})

    dcol = _find_col(race_results, "driver")
    pos_col = _find_col(race_results, "position", "win")

    if not dcol or not pos_col:
        return json.dumps({"error": "Required columns not found"})

    df_copy = race_results[[dcol, pos_col]].copy()
    df_copy["_win"] = pd.to_numeric(df_copy[pos_col], errors="coerce") == 1
    top = (
        df_copy.groupby(dcol)["_win"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    top.columns = ["driver", "wins"]
    top["wins"] = top["wins"].astype(int)
    top["rank"] = range(1, len(top) + 1)

    return json.dumps({"top_n": top_n, "drivers": top.to_dict(orient="records")})


# ── Chat function ──────────────────────────────────────────────────────────────
TOOLS = [get_driver_stats, get_champion_by_year, get_constructor_stats,
         search_race_results, get_top_drivers]

SYSTEM = (
    "You are an expert F1 Race Engineer AI with access to 70+ years of F1 historical data. "
    "Always use your tools to fetch accurate data before answering. "
    "Be enthusiastic, concise, and knowledgeable. Use F1 terminology naturally. "
    "When comparing drivers or teams, always back up claims with numbers from the data."
)


def execute_tool(name: str, args: dict) -> str:
    fn_map = {
        "get_driver_stats":      get_driver_stats,
        "get_champion_by_year":  get_champion_by_year,
        "get_constructor_stats": get_constructor_stats,
        "search_race_results":   search_race_results,
        "get_top_drivers":       get_top_drivers,
    }
    if name in fn_map:
        return fn_map[name](**args)
    return json.dumps({"error": f"Unknown tool: {name}"})


def chat(user_message: str, history: list, api_key: str) -> str:
    client = genai.Client(api_key=api_key)

    # Build full conversation as a contents list
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=msg["content"])]))
    contents.append(types.Content(role="user", parts=[types.Part(text=user_message)]))

    config = types.GenerateContentConfig(system_instruction=SYSTEM, tools=TOOLS)

    # Agentic loop — keep calling until no more function calls
    for _ in range(5):
        response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=contents,
            config=config,
        )

        candidate = response.candidates[0]
        fn_calls = [p.function_call for p in candidate.content.parts if p.function_call]

        if not fn_calls:
            return response.text

        # Add model response to contents
        contents.append(candidate.content)

        # Execute each function call and feed results back
        result_parts = []
        for fc in fn_calls:
            result = execute_tool(fc.name, dict(fc.args))
            result_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=fc.name,
                        response={"result": result},
                    )
                )
            )
        contents.append(types.Content(role="user", parts=result_parts))

    return response.text


# ── Session state ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# ── Display chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask me anything about F1..."):
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Checking the data..."):
                try:
                    reply = chat(prompt, st.session_state.history, api_key)
                    st.markdown(reply)
                    st.session_state.history.append({"role": "user", "content": prompt})
                    st.session_state.history.append({"role": "assistant", "content": reply})
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"Error: {e}")

# ── Suggested questions ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("**Quick questions — click to ask:**")

SUGGESTIONS = [
    "Who has the most F1 wins of all time?",
    "Who won the 2021 F1 World Championship?",
    "Which team dominated the 2010s?",
    "Compare Lewis Hamilton and Michael Schumacher",
    "What are Max Verstappen's career stats?",
]

cols = st.columns(len(SUGGESTIONS))
for col, suggestion in zip(cols, SUGGESTIONS):
    with col:
        if st.button(suggestion, use_container_width=True):
            if not api_key:
                st.error("Please enter your Gemini API key in the sidebar.")
            else:
                st.session_state.messages.append({"role": "user", "content": suggestion})
                with st.chat_message("user"):
                    st.markdown(suggestion)

                with st.chat_message("assistant"):
                    with st.spinner("Checking the data..."):
                        try:
                            reply = chat(suggestion, st.session_state.history, api_key)
                            st.markdown(reply)
                            st.session_state.history.append({"role": "user", "content": suggestion})
                            st.session_state.history.append({"role": "assistant", "content": reply})
                            st.session_state.messages.append({"role": "assistant", "content": reply})
                        except Exception as e:
                            st.error(f"Error: {e}")
                st.rerun()
