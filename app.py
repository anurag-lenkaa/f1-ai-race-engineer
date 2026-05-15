import streamlit as st
import anthropic
import pandas as pd
import json
import os

st.set_page_config(page_title="AI Race Engineer", page_icon="🔧", layout="wide")
st.title("🔧 AI Race Engineer")
st.markdown(
    "Chat with an F1 expert powered by Claude — it queries real race data to answer your questions."
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ── Sidebar ────────────────────────────────────────────────────────────────────
api_key = st.sidebar.text_input(
    "Anthropic API Key",
    type="password",
    help="Get yours at console.anthropic.com",
)
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
    st.warning(f"Could not load data files: {e}. Some tool responses may be limited.")
    race_results = pd.DataFrame()
    driver_champions = pd.DataFrame()
    constructor_champions = pd.DataFrame()
    data_loaded = False

# ── Tool definitions ───────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "get_driver_stats",
        "description": (
            "Get career statistics for an F1 driver including wins, championships, "
            "nationality, and teams raced for."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "driver_name": {
                    "type": "string",
                    "description": "Full or partial name of the driver (e.g. 'Hamilton', 'Lewis Hamilton')",
                }
            },
            "required": ["driver_name"],
        },
    },
    {
        "name": "get_champion_by_year",
        "description": "Get the F1 World Drivers' Champion and Constructors' Champion for a given year.",
        "input_schema": {
            "type": "object",
            "properties": {
                "year": {
                    "type": "integer",
                    "description": "The season year (e.g. 2021)",
                }
            },
            "required": ["year"],
        },
    },
    {
        "name": "get_constructor_stats",
        "description": (
            "Get career statistics for an F1 constructor/team including total wins "
            "and seasons active."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "constructor_name": {
                    "type": "string",
                    "description": "Full or partial name of the constructor (e.g. 'Ferrari', 'Red Bull')",
                }
            },
            "required": ["constructor_name"],
        },
    },
    {
        "name": "search_race_results",
        "description": (
            "Search race results across race name, driver name, constructor name, "
            "or country. Returns matching rows."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term to match against race, driver, constructor, or country fields",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_top_drivers",
        "description": "Get the top N F1 drivers ranked by total career wins.",
        "input_schema": {
            "type": "object",
            "properties": {
                "top_n": {
                    "type": "integer",
                    "description": "Number of top drivers to return (default 10)",
                    "default": 10,
                }
            },
            "required": [],
        },
    },
]


# ── Tool execution ─────────────────────────────────────────────────────────────
def _find_driver_col(df, hint="driver"):
    """Return the column name that most likely contains driver names."""
    for col in df.columns:
        if hint in col.lower():
            return col
    return None


def _find_constructor_col(df, hint="constructor"):
    for col in df.columns:
        if hint in col.lower() or "team" in col.lower():
            return col
    return None


def get_driver_stats(driver_name: str) -> dict:
    if race_results.empty:
        return {"error": "Race results data not available"}

    dcol = _find_driver_col(race_results)
    if dcol is None:
        return {"error": "Driver column not found in race_results"}

    mask = race_results[dcol].str.contains(driver_name, case=False, na=False)
    sub = race_results[mask]

    if sub.empty:
        return {"message": f"No results found for driver '{driver_name}'"}

    # Try to count wins
    result: dict = {"driver_name": driver_name, "races_found": len(sub)}

    win_col = next(
        (c for c in sub.columns if "win" in c.lower() or "position" in c.lower()), None
    )
    if win_col:
        try:
            wins = (pd.to_numeric(sub[win_col], errors="coerce") == 1).sum()
            result["wins"] = int(wins)
        except Exception:
            pass

    # Championships from driver_champions
    if not driver_champions.empty:
        dchamp_col = _find_driver_col(driver_champions)
        if dchamp_col:
            champ_mask = driver_champions[dchamp_col].str.contains(
                driver_name, case=False, na=False
            )
            championships = driver_champions[champ_mask]
            result["championships"] = len(championships)
            year_col = next(
                (c for c in championships.columns if "year" in c.lower() or "season" in c.lower()),
                None,
            )
            if year_col:
                result["championship_years"] = sorted(championships[year_col].tolist())

    # Teams
    ccol = _find_constructor_col(sub)
    if ccol:
        result["teams"] = sub[ccol].dropna().unique().tolist()

    # Nationality
    nat_col = next((c for c in sub.columns if "nation" in c.lower() or "country" in c.lower()), None)
    if nat_col:
        result["nationality"] = sub[nat_col].mode().iloc[0] if not sub[nat_col].empty else None

    return result


def get_champion_by_year(year: int) -> dict:
    result: dict = {"year": year}

    if not driver_champions.empty:
        year_col = next(
            (c for c in driver_champions.columns if "year" in c.lower() or "season" in c.lower()),
            None,
        )
        if year_col:
            row = driver_champions[driver_champions[year_col] == year]
            if not row.empty:
                dcol = _find_driver_col(driver_champions)
                result["drivers_champion"] = row[dcol].iloc[0] if dcol else row.iloc[0].to_dict()

    if not constructor_champions.empty:
        year_col = next(
            (
                c
                for c in constructor_champions.columns
                if "year" in c.lower() or "season" in c.lower()
            ),
            None,
        )
        if year_col:
            row = constructor_champions[constructor_champions[year_col] == year]
            if not row.empty:
                ccol = _find_constructor_col(constructor_champions)
                result["constructors_champion"] = (
                    row[ccol].iloc[0] if ccol else row.iloc[0].to_dict()
                )

    if len(result) == 1:
        result["message"] = f"No championship data found for {year}"

    return result


def get_constructor_stats(constructor_name: str) -> dict:
    if race_results.empty:
        return {"error": "Race results data not available"}

    ccol = _find_constructor_col(race_results)
    if ccol is None:
        return {"error": "Constructor column not found in race_results"}

    mask = race_results[ccol].str.contains(constructor_name, case=False, na=False)
    sub = race_results[mask]

    if sub.empty:
        return {"message": f"No results found for constructor '{constructor_name}'"}

    result: dict = {"constructor_name": constructor_name, "races_found": len(sub)}

    win_col = next(
        (c for c in sub.columns if "win" in c.lower() or "position" in c.lower()), None
    )
    if win_col:
        try:
            wins = (pd.to_numeric(sub[win_col], errors="coerce") == 1).sum()
            result["wins"] = int(wins)
        except Exception:
            pass

    year_col = next(
        (c for c in sub.columns if "year" in c.lower() or "season" in c.lower()), None
    )
    if year_col:
        years = pd.to_numeric(sub[year_col], errors="coerce").dropna()
        result["seasons_active"] = sorted(years.unique().astype(int).tolist())
        result["first_season"] = int(years.min())
        result["last_season"] = int(years.max())

    return result


def search_race_results(query: str) -> dict:
    if race_results.empty:
        return {"error": "Race results data not available"}

    str_cols = race_results.select_dtypes(include="object").columns
    mask = race_results[str_cols].apply(
        lambda col: col.str.contains(query, case=False, na=False)
    ).any(axis=1)

    matches = race_results[mask].head(20)

    if matches.empty:
        return {"message": f"No race results found matching '{query}'"}

    return {
        "query": query,
        "count": int(mask.sum()),
        "results": matches.to_dict(orient="records"),
    }


def get_top_drivers(top_n: int = 10) -> dict:
    if race_results.empty:
        return {"error": "Race results data not available"}

    dcol = _find_driver_col(race_results)
    win_col = next(
        (c for c in race_results.columns if "win" in c.lower() or "position" in c.lower()), None
    )

    if dcol is None or win_col is None:
        return {"error": "Required columns not found"}

    df_copy = race_results[[dcol, win_col]].copy()
    df_copy["_is_win"] = pd.to_numeric(df_copy[win_col], errors="coerce") == 1
    top = (
        df_copy.groupby(dcol)["_is_win"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )
    top.columns = ["driver", "wins"]
    top["wins"] = top["wins"].astype(int)
    top["rank"] = range(1, len(top) + 1)

    return {"top_n": top_n, "drivers": top.to_dict(orient="records")}


def execute_tool(name: str, inputs: dict) -> dict:
    if name == "get_driver_stats":
        return get_driver_stats(inputs["driver_name"])
    elif name == "get_champion_by_year":
        return get_champion_by_year(inputs["year"])
    elif name == "get_constructor_stats":
        return get_constructor_stats(inputs["constructor_name"])
    elif name == "search_race_results":
        return search_race_results(inputs["query"])
    elif name == "get_top_drivers":
        return get_top_drivers(inputs.get("top_n", 10))
    else:
        return {"error": f"Unknown tool: {name}"}


# ── Chat function ──────────────────────────────────────────────────────────────
def chat(user_message: str, history: list, api_key: str):
    client = anthropic.Anthropic(api_key=api_key)

    SYSTEM = (
        "You are an expert F1 Race Engineer AI with access to 70+ years of F1 data. "
        "Use your tools to retrieve accurate data before answering. "
        "Be enthusiastic, concise, and knowledgeable. Use F1 terminology naturally. "
        "When comparing drivers or teams, always back up claims with numbers from the data."
    )

    history = history + [{"role": "user", "content": user_message}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=SYSTEM,
            tools=TOOLS,
            messages=history,
        )

        if response.stop_reason == "tool_use":
            history.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": json.dumps(result, default=str),
                        }
                    )
            history.append({"role": "user", "content": tool_results})
        else:
            reply = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            history.append({"role": "assistant", "content": reply})
            return reply, history


# ── Session state init ─────────────────────────────────────────────────────────
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
        st.error("Please enter your Anthropic API key in the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Checking the data..."):
                try:
                    reply, st.session_state.history = chat(
                        prompt, st.session_state.history, api_key
                    )
                    st.markdown(reply)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": reply}
                    )
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
                st.error("Please enter your Anthropic API key in the sidebar.")
            else:
                st.session_state.messages.append(
                    {"role": "user", "content": suggestion}
                )
                with st.chat_message("user"):
                    st.markdown(suggestion)

                with st.chat_message("assistant"):
                    with st.spinner("Checking the data..."):
                        try:
                            reply, st.session_state.history = chat(
                                suggestion, st.session_state.history, api_key
                            )
                            st.markdown(reply)
                            st.session_state.messages.append(
                                {"role": "assistant", "content": reply}
                            )
                        except Exception as e:
                            st.error(f"Error: {e}")
                st.rerun()
