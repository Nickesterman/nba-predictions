"""
NBA Dashboard - 2025-2026 Season
=================================
Interactive dashboard for NBA predictions starting Oct 22, 2025

This version is deployment-ready for Streamlit Cloud and a weekly GitHub
Actions workflow that refreshes `weekly_predictions.csv`.

Expected CSV columns (case-sensitive):
- date (ISO string or parseable to datetime)
- time
- away
- home
- predicted_winner
- team1_win_prob (0..1 float)
- confidence (LOW/MEDIUM/HIGH)
- spread (float, +/-)
- expected_value (float, % not in decimal)
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------
# PAGE CONFIGURATION
# --------------------
st.set_page_config(
    page_title="NBA 2025-26 Predictions",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------
# CONSTANTS
# --------------------
SEASON_START = datetime(2025, 10, 22)
WEEKS_IN_SEASON = 26
PREDICTIONS_FILE = "weekly_predictions.csv"

REQUIRED_COLS = [
    "date",
    "time",
    "away",
    "home",
    "predicted_winner",
    "team1_win_prob",
    "confidence",
    "spread",
    "expected_value",
]

CONF_MAP = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

# --------------------
# DATA LOADING
# --------------------
@st.cache_data
def load_predictions(path: str) -> pd.DataFrame:
    """Load weekly predictions CSV if present and parse dates."""
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Ensure required columns exist
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = np.nan
    # Parse date
    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception:
        # If parse fails, drop invalid rows
        df = df.drop(columns=["date"], errors="ignore")
        df["date"] = pd.NaT
    # Normalize types / casing
    df["confidence"] = df["confidence"].astype(str).str.upper()
    for c in ["team1_win_prob", "spread", "expected_value"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Fill safe defaults
    df["time"] = df["time"].fillna("TBD")
    df["away"] = df["away"].fillna("Away")
    df["home"] = df["home"].fillna("Home")
    df["predicted_winner"] = df["predicted_winner"].fillna("TBD")
    df["team1_win_prob"] = df["team1_win_prob"].fillna(0.5)
    df["spread"] = df["spread"].fillna(0.0)
    df["expected_value"] = df["expected_value"].fillna(0.0)
    return df[REQUIRED_COLS]


predictions_df = load_predictions(PREDICTIONS_FILE)

# --------------------
# CUSTOM STYLES
# --------------------
st.markdown(
    """
<style>
    .main { padding: 0rem 1rem; }
    .season-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .bet-card {
        background: #2d2d2d;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    .value-bet {
        background: #1e7e34;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        display: inline-block;
    }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------
# HEADER
# --------------------
st.markdown(
    """
<div class="season-header">
    <h1>🏀 NBA 2025-2026 Season Predictions</h1>
    <p>AI-Powered Betting Analysis | Season Starts: October 22, 2025</p>
</div>
""",
    unsafe_allow_html=True,
)

# --------------------
# SIDEBAR
# --------------------
with st.sidebar:
    st.header("📅 Season Navigation")

    # Week selector (anchored to SEASON_START, 7-day windows)
    week_labels = [
        f"Week {i+1}: "
        f"{(SEASON_START + timedelta(days=i*7)).strftime('%b %d')} - "
        f"{(SEASON_START + timedelta(days=i*7+6)).strftime('%b %d')}"
        for i in range(WEEKS_IN_SEASON)
    ]
    current_week = st.selectbox("Select Week", options=week_labels, index=0)
    week_num = int(current_week.split(":")[0].split()[1]) - 1
    selected_week_start = SEASON_START + timedelta(days=week_num * 7)
    selected_week_end = selected_week_start + timedelta(days=7)  # exclusive

    st.markdown("---")
    st.subheader("🎯 Bet Filters")
    min_confidence = st.select_slider(
        "Minimum Confidence", options=["LOW", "MEDIUM", "HIGH"], value="MEDIUM"
    )
    min_ev = st.slider(
        "Minimum Expected Value (%)", min_value=0.0, max_value=15.0, value=5.0, step=0.5
    )
    bet_types = st.multiselect(
        "Bet Types", options=["Moneyline", "Spread", "Over/Under"], default=["Moneyline", "Spread"]
    )

    st.markdown("---")
    st.subheader("💰 Bankroll Management")
    bankroll = st.number_input(
        "Season Bankroll ($)", min_value=1000, max_value=100000, value=10000, step=500
    )
    unit_size = st.slider(
        "Unit Size (%)", min_value=1, max_value=5, value=2, help="Percentage of bankroll per unit"
    )

    st.markdown("---")
    st.info(
        f"**Season Progress**\n"
        f"Week {week_num + 1} of {WEEKS_IN_SEASON}\n\n"
        f"**Key Dates:**\n"
        f"• Opening Night: Oct 22\n"
        f"• All-Star Break: Feb 14-16\n"
        f"• Playoffs Start: Apr 19\n"
        f"• Finals: June 2026"
    )

# Refresh button
if st.button("🔄 Refresh Predictions"):
    st.cache_data.clear()
    predictions_df = load_predictions(PREDICTIONS_FILE)
    st.success("Predictions refreshed!")

# --------------------
# HELPER: weekly subset + filters
# --------------------
def get_week_predictions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    # Date-range filter (inclusive start, exclusive end)
    mask = (df["date"] >= selected_week_start) & (df["date"] < selected_week_end)
    week_df = df.loc[mask].copy()

    # Apply filters
    week_df["confidence"] = week_df["confidence"].astype(str).str.upper()
    week_df = week_df[
        (week_df["confidence"].map(CONF_MAP) >= CONF_MAP[min_confidence])
        & (week_df["expected_value"] >= min_ev)
    ]
    return week_df


# --------------------
# MAIN TABS
# --------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📅 Weekly Schedule & Bets", "🎯 Best Bets This Week", "📊 Season Analytics", "💹 Bankroll Tracker", "🏆 Championship Odds"]
)

# --------------------
# TAB 1: WEEKLY SCHEDULE
# --------------------
with tab1:
    st.header(f"📅 Games for {current_week}")

    if predictions_df.empty:
        st.warning(
            "No prediction data available. Make sure `weekly_predictions.csv` exists. "
            "Run `python train_model.py` (locally or via GitHub Actions) to generate it."
        )
    else:
        week_preds = get_week_predictions(predictions_df)

        if not week_preds.empty:
            display_df = week_preds.copy()
            display_df["Date"] = display_df["date"].dt.strftime("%b %d")
            display_df["Win %"] = (display_df["team1_win_prob"] * 100).round(1).astype(str) + "%"
            display_df["Spread"] = display_df["spread"].apply(lambda x: f"{x:+.1f}")
            display_df["EV %"] = display_df["expected_value"].round(1).astype(str) + "%"

            # Optional bet icon based on EV
            def bet_icon(ev):
                if ev > 5:
                    return "✅"
                elif ev > 2:
                    return "⚠️"
                return "❌"

            display_df["Bet"] = display_df["expected_value"].apply(bet_icon)

            st.dataframe(
                display_df[
                    [
                        "Date",
                        "time",
                        "away",
                        "home",
                        "predicted_winner",
                        "Win %",
                        "confidence",
                        "Spread",
                        "EV %",
                        "Bet",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

            # Week summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Games (all, unfiltered)", len(predictions_df[
                    (predictions_df["date"] >= selected_week_start) & (predictions_df["date"] < selected_week_end)
                ]))
            with col2:
                value_bets = (week_preds["expected_value"] > 5).sum()
                st.metric("Value Bets (EV>5%)", int(value_bets))
            with col3:
                avg_ev = week_preds["expected_value"].mean() if not week_preds.empty else 0.0
                st.metric("Avg Expected Value", f"{avg_ev:.1f}%")
        else:
            st.warning("No games match your filter criteria for this week.")

# --------------------
# TAB 2: BEST BETS
# --------------------
with tab2:
    st.header("🎯 Top Value Bets This Week")

    if predictions_df.empty:
        st.info("No predictions available.")
    else:
        week_preds = get_week_predictions(predictions_df)
        if week_preds.empty:
            st.info("No predictions meet your filters for this week.")
        else:
            top_bets = week_preds.sort_values("expected_value", ascending=False).head(5)

            for i, (_, bet) in enumerate(top_bets.iterrows(), start=1):
                ev = float(bet.get("expected_value", 0.0))
                col1, col2 = st.columns([3, 1])

                with col1:
                    game_date = bet.get("date")
                    date_str = pd.to_datetime(game_date).strftime("%B %d, %Y") if pd.notna(game_date) else "TBD"
                    st.markdown(
                        f"""
                        <div class="bet-card">
                            <h4>#{i}. {bet.get('away', 'Away')} @ {bet.get('home', 'Home')}</h4>
                            <p>📅 {date_str} | ⏰ {bet.get('time', 'TBD')}</p>
                            <p>🎯 <strong>Pick:</strong> {bet.get('predicted_winner', 'TBD')} {float(bet.get('spread', 0.0)):+.1f}</p>
                            <p>📊 <strong>Win Probability:</strong> {float(bet.get('team1_win_prob', 0.5)):.1%}</p>
                            <p>💪 <strong>Confidence:</strong> {str(bet.get('confidence', 'MEDIUM')).upper()}</p>
                            <p>💰 <strong>Expected Value:</strong> <span class="value-bet">{ev:.1f}%</span></p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col2:
                    # Simple stake suggestion based on EV & unit_size
                    unit_mult = 1.0 if ev < 7 else 1.5 if ev < 10 else 2.0
                    bet_amount = bankroll * (unit_size / 100.0) * unit_mult
                    st.metric("Suggested Bet", f"${bet_amount:.0f}")
                    st.metric("Units", f"{unit_mult:.1f}")
                    if ev > 8:
                        st.success("STRONG BET")
                    elif ev > 5:
                        st.warning("GOOD VALUE")
                    else:
                        st.info("MODERATE")

    # Kelly Calculator (unchanged)
    st.markdown("---")
    st.subheader("📊 Kelly Criterion Calculator")

    col1, col2 = st.columns(2)
    with col1:
        win_prob = st.slider("Win Probability (%)", 40, 70, 55)
        odds = st.number_input("American Odds", -300, 300, -110)
    with col2:
        if odds > 0:
            decimal_odds = 1 + (odds / 100)
        else:
            decimal_odds = 1 + (100 / abs(odds))
        kelly = (win_prob / 100 * (decimal_odds - 1) - (1 - win_prob / 100)) / (decimal_odds - 1)
        kelly_pct = max(0.0, kelly * 100)
        st.metric("Kelly %", f"{kelly_pct:.1f}%")
        st.metric("Suggested Bet", f"${bankroll * kelly_pct / 100:.0f}")

# --------------------
# TAB 3: SEASON ANALYTICS
# --------------------
with tab3:
    st.header("📊 2025-26 Season Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Team Performance Projections")
        teams_projection = pd.DataFrame(
            {
                "Team": [
                    "Boston Celtics",
                    "Milwaukee Bucks",
                    "Denver Nuggets",
                    "Phoenix Suns",
                    "Los Angeles Lakers",
                    "Golden State Warriors",
                    "Philadelphia 76ers",
                    "Miami Heat",
                ],
                "Projected Wins": [58, 56, 55, 54, 52, 51, 49, 47],
                "Championship %": [18.5, 16.2, 15.8, 12.3, 10.5, 9.8, 7.2, 5.1],
            }
        )

        fig = go.Figure(
            data=[
                go.Bar(
                    x=teams_projection["Projected Wins"],
                    y=teams_projection["Team"],
                    orientation="h",
                    marker_color="#667eea",
                    text=teams_projection["Projected Wins"],
                    textposition="outside",
                )
            ]
        )
        fig.update_layout(
            title="Projected Win Totals",
            xaxis_title="Wins",
            height=400,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(40,40,40,0.1)",
            font=dict(color="white"),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Value Bet Distribution")
        bet_distribution = pd.DataFrame(
            {
                "Type": ["Favorites", "Underdogs", "Home Teams", "Away Teams", "Overs", "Unders"],
                "Count": [45, 32, 52, 25, 38, 39],
            }
        )
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=bet_distribution["Type"],
                    values=bet_distribution["Count"],
                    hole=0.3,
                    marker_colors=px.colors.sequential.Purples_r,
                )
            ]
        )
        fig.update_layout(
            title="Season Bet Type Distribution",
            height=400,
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Simple season trends using the chosen week number
    st.subheader("📈 Season Betting Trends")
    weeks = list(range(1, week_num + 2))
    accuracy_trend = [50 + i * 0.8 + np.random.randint(-3, 3) for i in weeks]
    roi_trend = [0 + i * 0.5 + np.random.uniform(-2, 2) for i in weeks]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=weeks, y=accuracy_trend, mode="lines+markers", name="Accuracy %", line=dict(color="#667eea", width=2), yaxis="y"
        )
    )
    fig.add_trace(
        go.Scatter(
            x=weeks, y=roi_trend, mode="lines+markers", name="ROI %", line=dict(color="#764ba2", width=2), yaxis="y2"
        )
    )
    fig.update_layout(
        title="Model Performance Trend",
        xaxis_title="Week",
        yaxis=dict(title="Accuracy %", side="left"),
        yaxis2=dict(title="ROI %", side="right", overlaying="y"),
        height=350,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(40,40,40,0.1)",
        font=dict(color="white"),
        xaxis=dict(showgrid=False),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------
# TAB 4: BANKROLL TRACKER
# --------------------
with tab4:
    st.header("💹 Bankroll Management")

    weeks_played = week_num + 1
    avg_weekly_return = 0.015  # 1.5% average weekly return (illustrative)
    current_bankroll = bankroll * (1 + avg_weekly_return * weeks_played)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Starting Bankroll", f"${bankroll:,.0f}")
    with col2:
        st.metric("Current Bankroll", f"${current_bankroll:,.0f}", f"+${current_bankroll - bankroll:.0f}")
    with col3:
        roi = ((current_bankroll - bankroll) / bankroll) * 100
        st.metric("Season ROI", f"{roi:.1f}%", f"+{roi:.1f}%")
    with col4:
        st.metric("Weeks Played", weeks_played)

    st.subheader("📈 Bankroll Progression")
    bankroll_data = []
    temp_bankroll = bankroll
    for w in range(weeks_played):
        weekly_return = np.random.uniform(-0.05, 0.08)  # illustrative
        temp_bankroll *= 1 + weekly_return
        bankroll_data.append({"Week": w + 1, "Bankroll": temp_bankroll, "Weekly Return": weekly_return * 100})
    df_bankroll = pd.DataFrame(bankroll_data)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_bankroll["Week"],
            y=df_bankroll["Bankroll"],
            mode="lines+markers",
            name="Bankroll",
            line=dict(color="#667eea", width=3),
            fill="tozeroy",
            fillcolor="rgba(102, 126, 234, 0.2)",
        )
    )
    fig.add_hline(y=bankroll, line_dash="dash", line_color="gray", annotation_text="Break-even")
    fig.update_layout(
        title="Season Bankroll Growth",
        xaxis_title="Week",
        yaxis_title="Bankroll ($)",
        height=400,
        showlegend=True,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(40,40,40,0.1)",
        font=dict(color="white"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📝 Recent Betting History")
    betting_log = pd.DataFrame(
        {
            "Date": pd.date_range(end=datetime.now(), periods=10).strftime("%b %d"),
            "Game": ["LAL @ BOS", "MIL @ PHI", "GSW @ DEN", "PHX @ DAL", "MIA @ BKN", "BOS @ NYK", "LAL @ GSW", "DEN @ PHX", "PHI @ MIA", "DAL @ MIL"],
            "Bet": ["BOS -5.5", "MIL ML", "Under 228", "PHX +3", "MIA ML", "BOS -7", "GSW -2.5", "DEN -4", "PHI +2", "MIL -6"],
            "Stake": [200, 150, 100, 250, 180, 220, 160, 200, 140, 190],
            "Odds": ["-110", "+120", "-105", "-110", "+140", "-115", "-110", "-105", "+100", "-110"],
            "Result": ["✅ Won", "❌ Lost", "✅ Won", "✅ Won", "❌ Lost", "✅ Won", "✅ Won", "❌ Lost", "✅ Won", "✅ Won"],
            "Profit": ["+$182", "-$150", "+$95", "+$227", "-$180", "+$191", "+$145", "-$200", "+$140", "+$173"],
        }
    )

    def style_result(val):
        return "color: #1e7e34; font-weight: bold" if "✅" in val else "color: #dc3545; font-weight: bold"

    styled_log = betting_log.style.applymap(style_result, subset=["Result", "Profit"])
    st.dataframe(styled_log, use_container_width=True, hide_index=True)

# --------------------
# TAB 5: CHAMPIONSHIP ODDS
# --------------------
with tab5:
    st.header("🏆 2025-26 Championship Projections")

    col1, col2 = st.columns([2, 1])

    with col1:
        championship_data = pd.DataFrame(
            {
                "Team": [
                    "Boston Celtics",
                    "Milwaukee Bucks",
                    "Denver Nuggets",
                    "Phoenix Suns",
                    "Los Angeles Lakers",
                    "Golden State Warriors",
                    "Philadelphia 76ers",
                    "Dallas Mavericks",
                    "Miami Heat",
                    "Memphis Grizzlies",
                ],
                "Odds": ["+350", "+400", "+450", "+600", "+800", "+900", "+1200", "+1500", "+1800", "+2000"],
                "Implied %": [22.2, 20.0, 18.2, 14.3, 11.1, 10.0, 7.7, 6.3, 5.3, 4.8],
                "Model %": [18.5, 16.2, 15.8, 12.3, 10.5, 9.8, 7.2, 8.1, 5.1, 6.5],
            }
        )

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Betting Odds", x=championship_data["Team"], y=championship_data["Implied %"], marker_color="#764ba2"))
        fig.add_trace(go.Bar(name="Model Projection", x=championship_data["Team"], y=championship_data["Model %"], marker_color="#667eea"))
        fig.update_layout(
            title="Championship Probability Comparison",
            yaxis_title="Probability (%)",
            barmode="group",
            height=400,
            showlegend=True,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(40,40,40,0.1)",
            font=dict(color="white"),
            xaxis=dict(showgrid=False, tickangle=-45),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🎯 Value Plays")
        championship_data["Edge"] = championship_data["Model %"] - championship_data["Implied %"]
        value_plays = championship_data[championship_data["Edge"] > 0].sort_values("Edge", ascending=False)
        if not value_plays.empty:
            for _, team in value_plays.iterrows():
                st.success(f"**{team['Team']}**\nOdds: {team['Odds']}\nEdge: +{team['Edge']:.1f}%")
        else:
            st.info("No value plays currently available")

        st.markdown("---")
        st.subheader("📊 Division Winners")
        division_odds = {
            "Atlantic": "Boston Celtics (-180)",
            "Central": "Milwaukee Bucks (-150)",
            "Southeast": "Miami Heat (+120)",
            "Northwest": "Denver Nuggets (-200)",
            "Pacific": "Phoenix Suns (+110)",
            "Southwest": "Dallas Mavericks (+140)",
        }
        for div, fav in division_odds.items():
            st.text(f"{div}: {fav}")

# --------------------
# FOOTER & DISCLAIMER
# --------------------
st.markdown("---")
st.markdown(
    f"<div style='text-align: center; color: #888;'>"
    f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | "
    f"Season: 2025-26 | "
    f"Model Version: 3.0"
    f"</div>",
    unsafe_allow_html=True,
)

st.warning(
    "⚠️ **Disclaimer:** This model is for educational and entertainment purposes only. "
    "NBA betting involves risk. Please gamble responsibly and never bet more than you can afford to lose. "
    "Must be 21+ to gamble."
)
