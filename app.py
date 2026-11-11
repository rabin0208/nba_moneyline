"""Daily slate and holdout profit, laid out like the baseball moneyline app."""

from __future__ import annotations

import math
import sys
from datetime import date, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nba_moneyline.market import (  # noqa: E402
    SLATE_CAP_FRAC,
    american_to_decimal,
    cap_slate_stakes,
    format_american_odds,
    pick_bets,
    prepare_profit_bets,
    recommended_bankroll_frac,
)
from nba_moneyline.closing_odds import CLOSING_ODDS_CSV  # noqa: E402
from nba_moneyline.profits import available_seasons, load_model_games, season_label  # noqa: E402
from nba_moneyline.train import FEATURES_CSV  # noqa: E402
from nba_moneyline.slate import score_slate  # noqa: E402

st.set_page_config(page_title="NBA Moneyline", page_icon="🏀", layout="wide")

st.markdown(
    """
    <style>
    :root {
      --nba-bg: #10140f;
      --nba-panel: #171d14;
      --nba-line: #31402a;
      --nba-text: #f4f7ef;
      --nba-muted: #b7c3ad;
      --nba-accent: #f58426;
      --nba-green: #7dce6a;
    }
    .stApp {
      background:
        radial-gradient(circle at 80% 0%, rgba(245, 132, 38, 0.16), transparent 24rem),
        linear-gradient(180deg, #1c2418 0, var(--nba-bg) 22rem);
      color: var(--nba-text);
    }
    [data-testid="stSidebar"] {
      background: rgba(23, 29, 20, 0.98);
      border-right: 1px solid var(--nba-line);
    }
    [data-testid="stMetric"] {
      background: rgba(23, 29, 20, 0.82);
      border: 1px solid var(--nba-line);
      border-radius: 0.75rem;
      padding: 0.85rem 1rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
      background: rgba(23, 29, 20, 0.76);
      border-color: var(--nba-line);
      border-radius: 0.8rem;
    }
    .nba-hero {
      margin: 0 0 1.4rem;
      padding: clamp(1.2rem, 3vw, 2.1rem);
      background: linear-gradient(100deg, #c45012 0%, #f58426 58%, #ffd29a 140%);
      border-radius: 0.8rem;
    }
    .nba-hero h1 { margin: 0; color: white; font-size: clamp(2.1rem, 5vw, 3.6rem); line-height: 1; }
    .nba-hero p { margin: 0.55rem 0 0; color: rgba(255,255,255,0.9); font-size: 1.05rem; }
    .nba-game-title { color: var(--nba-text); font-size: 1.2rem; font-weight: 750; }
    .nba-meta { color: var(--nba-muted); font-size: 0.83rem; }
    .nba-pick, .nba-stake {
      display: inline-block;
      margin: 0.15rem 0.5rem 0.8rem 0;
      padding: 0.28rem 0.6rem;
      border-radius: 0.3rem;
      font-size: 0.76rem;
      font-weight: 800;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }
    .nba-pick { color: #10210c; background: var(--nba-green); }
    .nba-stake { color: #2a1604; background: var(--nba-accent); }
    .nba-no-pick { color: var(--nba-muted); font-size: 0.8rem; }
    h1, h2, h3, label, [data-testid="stMetricLabel"] { color: var(--nba-text) !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_games(features_mtime: float, odds_mtime: float) -> pd.DataFrame:
    return load_model_games()


@st.cache_data(ttl=30 * 60, show_spinner=False)
def cached_slate(iso_date: str) -> pd.DataFrame:
    return score_slate(date.fromisoformat(iso_date))


def dollar_delta(units: float, unit_dollars: float) -> str:
    amount = units * unit_dollars
    if amount < 0:
        return f"-${abs(amount):,.0f}"
    return f"+${amount:,.0f}"


def dollars(amount: float) -> str:
    if amount < 0:
        return f"-${abs(amount):,.0f}"
    return f"${amount:,.0f}"


def percent(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.1%}"


def moneyline(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return format_american_odds(float(value))


def monthly_table(bets: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Month",
        "Bets",
        "Wins",
        "Hit rate",
        "Flat profit (u)",
        "Flat ROI",
        "Kelly profit (u)",
        "Kelly ROI",
    ]
    if bets.empty:
        return pd.DataFrame(columns=columns)
    work = bets.copy()
    work["month"] = pd.to_datetime(work["game_date"]).dt.to_period("M")
    monthly = (
        work.groupby("month", as_index=False)
        .agg(
            Bets=("bet_profit", "count"),
            Wins=("bet_won", "sum"),
            flat_profit=("bet_profit", "sum"),
            kelly_wagered=("kelly_stake_u", "sum"),
            kelly_profit=("kelly_profit_u", "sum"),
        )
        .sort_values("month")
    )
    monthly["Wins"] = monthly["Wins"].astype(int)
    monthly["Hit rate"] = monthly["Wins"] / monthly["Bets"]
    monthly["Flat ROI"] = monthly["flat_profit"] / monthly["Bets"]
    monthly["Kelly ROI"] = monthly["kelly_profit"] / monthly["kelly_wagered"].where(
        monthly["kelly_wagered"] > 0
    )
    monthly["Month"] = monthly["month"].astype(str)
    monthly["Flat profit (u)"] = monthly["flat_profit"]
    monthly["Kelly profit (u)"] = monthly["kelly_profit"]
    return monthly[columns]


def cumulative_frame(bets: pd.DataFrame) -> pd.DataFrame:
    work = bets.copy()
    work["Flat profit"] = work["bet_profit"].cumsum()
    work["Kelly-weighted profit"] = work["kelly_profit_u"].cumsum()
    return work.melt(
        id_vars=["game_date"],
        value_vars=["Flat profit", "Kelly-weighted profit"],
        var_name="Strategy",
        value_name="Profit (u)",
    )


def bet_table(bets: pd.DataFrame, unit_dollars: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": bets["game_date"],
            "Matchup": bets["away_team"] + " @ " + bets["home_team"],
            "Bet": bets.apply(
                lambda row: (
                    f"{row['home_team'] if row['bet_side'] == 'home' else row['away_team']} "
                    f"{format_american_odds(row['bet_odds'])}"
                ),
                axis=1,
            ),
            "Edge": bets["bet_edge"].map(lambda edge: f"{edge:+.1%}"),
            "Result": bets["bet_won"].map(lambda won: "Win" if won else "Loss"),
            "Score": bets["away_score"].astype(int).astype(str) + "-" + bets["home_score"].astype(int).astype(str),
            "Profit (u)": bets["bet_profit"].map(lambda profit: f"{profit:+.2f}"),
            "Profit ($)": (bets["bet_profit"] * unit_dollars).map(
                lambda amount: f"-${abs(amount):,.0f}" if amount < 0 else f"+${amount:,.0f}"
            ),
        }
    )


def attach_stakes(games: pd.DataFrame) -> pd.DataFrame:
    out = games.copy()
    stakes: list[float] = []
    for row in out.itertuples(index=False):
        if pd.isna(row.bet_side) or pd.isna(row.bet_prob) or pd.isna(row.bet_odds):
            stakes.append(math.nan)
            continue
        market = row.p_home_mkt if row.bet_side == "home" else row.p_away_mkt
        stakes.append(recommended_bankroll_frac(row.bet_prob, market, row.bet_odds))
    out["stake_frac"] = stakes
    chosen = out["bet_side"].notna()
    if chosen.any():
        out.loc[chosen, "stake_frac"] = cap_slate_stakes(out.loc[chosen, "stake_frac"]).to_numpy()
    return out


def slate_profit_if_all_win(games: pd.DataFrame, bankroll: float) -> float:
    profit = 0.0
    for row in games.itertuples(index=False):
        frac = getattr(row, "stake_frac", math.nan)
        odds = getattr(row, "bet_odds", math.nan)
        if pd.isna(frac) or float(frac) <= 0 or pd.isna(odds):
            continue
        decimal = american_to_decimal(odds)
        if math.isnan(decimal) or decimal <= 1.0:
            continue
        stake = float(frac) * bankroll if bankroll > 0 else float(frac)
        profit += stake * (decimal - 1.0)
    return profit


def render_side(
    *,
    team: str,
    model_prob: object,
    market_prob: object,
    edge: object,
    odds: object,
    show_market: bool,
) -> None:
    st.subheader(team)
    st.metric(
        "Model win probability",
        percent(model_prob),
        delta=f"{float(edge):+.1%} margin" if show_market and pd.notna(edge) else None,
    )
    if pd.notna(model_prob):
        st.progress(min(1.0, max(0.0, float(model_prob))), text="Logistic regression")
    if show_market:
        st.metric("DraftKings fair probability", percent(market_prob))
        if pd.notna(market_prob):
            st.progress(min(1.0, max(0.0, float(market_prob))), text=f"Moneyline {moneyline(odds)}")


def render_daily_slate(
    *,
    target_date: date,
    is_past: bool,
    recommendations_only: bool,
    edge_threshold: float,
    bankroll: float,
) -> None:
    with st.spinner(f"Scoring the {target_date:%B %-d} slate…"):
        try:
            raw = cached_slate(target_date.isoformat())
        except Exception as exc:
            st.error(str(exc))
            return
    if raw.empty:
        st.info(
            "No regular-season games were found for this date."
            if is_past
            else "No upcoming regular-season games were found for this date."
        )
        return

    games = raw.copy()
    games["p_away_win"] = 1.0 - games["p_home_win"]
    if is_past:
        recommendations = games.iloc[0:0]
        matched = games.iloc[0:0]
        slate_stake = 0.0
    else:
        games = attach_stakes(pick_bets(games, edge_threshold=edge_threshold))
        games = games.sort_values(["bet_edge", "game_id"], ascending=[False, True], na_position="last")
        recommendations = games.loc[games["bet_side"].notna()]
        matched = games.dropna(subset=["home_odds", "away_odds"])
        slate_stake = float(recommendations["stake_frac"].fillna(0).sum()) if not recommendations.empty else 0.0

    kpis = st.columns(4)
    kpis[0].metric("Games", len(games))
    if is_past:
        kpis[1].metric("Priced", "—")
        kpis[2].metric("Recommended", "—")
        kpis[3].metric("Slate stake", "—")
    else:
        kpis[1].metric("Priced", f"{len(matched)}/{len(games)}")
        kpis[2].metric("Recommended", len(recommendations))
        if bankroll > 0:
            kpis[3].metric("Slate stake", dollars(slate_stake * bankroll), delta=percent(slate_stake))
        else:
            kpis[3].metric("Slate stake", percent(slate_stake))

    visible = recommendations if recommendations_only and not is_past else games
    if visible.empty:
        st.info(f"No bets clear a {edge_threshold:.0%} margin.")
        return

    st.subheader(f"{target_date:%A, %B %-d}")
    if is_past:
        st.caption(f"{len(visible)} games · model probabilities only")
    else:
        provider = next((str(name) for name in games["provider"] if isinstance(name, str) and name), "DraftKings")
        st.caption(
            f"{len(visible)} of {len(games)} games · margin is model probability minus the vig-free "
            f"{provider} line · recommendation threshold {edge_threshold:.0%} · "
            f"¼ Kelly on a 50/50 blend, max 5% per bet, {SLATE_CAP_FRAC:.0%} slate cap"
        )

    for _, game in visible.iterrows():
        with st.container(border=True):
            header, status = st.columns([4, 1])
            header.markdown(
                f"<div class='nba-game-title'>{game['away_name']} @ {game['home_name']}</div>",
                unsafe_allow_html=True,
            )
            detail = str(game.get("status_detail") or "")
            spread = str(game.get("spread") or "")
            extra = f"<br>Line {spread}" if spread else ""
            status.markdown(f"<div class='nba-meta'>{detail}{extra}</div>", unsafe_allow_html=True)

            if is_past:
                away_score = game.get("away_score")
                home_score = game.get("home_score")
                if pd.notna(away_score) and pd.notna(home_score):
                    st.caption(f"Final {int(away_score)}–{int(home_score)}")
            elif pd.notna(game.get("bet_side")):
                side = str(game["bet_side"])
                team = game["home_name"] if side == "home" else game["away_name"]
                frac = game.get("stake_frac")
                stake = ""
                if pd.notna(frac) and float(frac) > 0:
                    stake_text = f"Stake {float(frac):.1%}"
                    if bankroll > 0:
                        stake_text += f" · {dollars(float(frac) * bankroll)}"
                    stake = f"<span class='nba-stake'>{stake_text}</span>"
                st.markdown(
                    f"<span class='nba-pick'>Margin: {team} {moneyline(game['bet_odds'])} · "
                    f"{percent(game['bet_edge'])}</span>{stake}",
                    unsafe_allow_html=True,
                )
            elif pd.notna(game.get("home_odds")):
                st.markdown(
                    "<span class='nba-no-pick'>No side clears the selected margin.</span>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<span class='nba-no-pick'>No moneyline posted yet.</span>",
                    unsafe_allow_html=True,
                )

            if pd.isna(game.get("p_home_win")):
                st.caption("Not enough prior games to score this matchup.")
                continue
            away_col, home_col = st.columns(2)
            with away_col:
                render_side(
                    team=str(game["away_name"]),
                    model_prob=game["p_away_win"],
                    market_prob=game.get("p_away_mkt"),
                    edge=game.get("edge_away"),
                    odds=game.get("away_odds"),
                    show_market=not is_past,
                )
            with home_col:
                render_side(
                    team=str(game["home_name"]),
                    model_prob=game["p_home_win"],
                    market_prob=game.get("p_home_mkt"),
                    edge=game.get("edge_home"),
                    odds=game.get("home_odds"),
                    show_market=not is_past,
                )


def render_profit_tab() -> None:
    try:
        games = cached_games(FEATURES_CSV.stat().st_mtime, CLOSING_ODDS_CSV.stat().st_mtime)
    except Exception as exc:
        st.error(str(exc))
        return
    seasons = available_seasons(games)
    default_index = len(seasons) - 1
    controls = st.columns([1, 1, 1, 2])
    with controls[0]:
        season_start = st.selectbox(
            "Season",
            options=seasons,
            index=default_index,
            format_func=season_label,
        )
    with controls[1]:
        price = st.segmented_control(
            "Line",
            options=["Open", "Close"],
            default="Open",
            required=True,
            key="profit_line",
        )
    with controls[2]:
        unit_dollars = st.number_input("Unit size ($)", min_value=1, value=100, step=25)
    with controls[3]:
        edge_percent = st.slider(
            "Minimum edge",
            min_value=0,
            max_value=20,
            value=5,
            step=1,
            format="%d%%",
            key="profit_edge",
            help="Model probability minus the vig-free price of the selected line.",
        )
    season_games = games.loc[games["season_start"] == season_start].copy()
    use_open = price != "Close"
    if use_open:
        season_games = season_games.dropna(subset=["home_open", "away_open"]).copy()
        season_games["home_odds"] = season_games["home_open"]
        season_games["away_odds"] = season_games["away_open"]
    line_name = "open" if use_open else "close"
    bets = prepare_profit_bets(season_games, edge_percent / 100.0)
    provider = season_games["provider"].dropna().mode()
    provider_name = str(provider.iloc[0]) if not provider.empty else "DraftKings"
    st.caption(
        f"{season_label(season_start)} · {len(season_games):,} games with a {provider_name} {line_name}. "
        "Edge and profit both use that price. Flat stakes are 1 unit. "
        "Kelly stakes are rescaled so the average stake is also 1 unit."
    )

    kpis = st.columns(4)
    if bets.empty:
        kpis[0].metric("Bets", 0)
        kpis[1].metric("Hit rate", "—")
        kpis[2].metric("Flat profit", "—")
        kpis[3].metric("Kelly profit", "—")
        st.info(f"No bets clear a {edge_percent}% edge in {season_label(season_start)}.")
        return

    flat_profit = float(bets["bet_profit"].sum())
    kelly_profit = float(bets["kelly_profit_u"].sum())
    hit_rate = float(bets["bet_won"].mean())
    flat_roi = flat_profit / len(bets)
    kelly_roi = kelly_profit / float(bets["kelly_stake_u"].sum())
    kpis[0].metric("Bets", len(bets))
    kpis[1].metric("Hit rate", f"{hit_rate:.1%}")
    kpis[2].metric("Flat profit", f"{flat_profit:+.1f} u", delta=dollar_delta(flat_profit, float(unit_dollars)))
    kpis[3].metric(
        "Kelly profit",
        f"{kelly_profit:+.1f} u",
        delta=f"{kelly_roi - flat_roi:+.1%} vs flat",
    )
    if kelly_roi < flat_roi - 0.005:
        st.warning(
            "Kelly-weighted ROI is lower: the biggest edges lost more than the smaller ones. "
            "That is overconfidence. The close was the sharper number."
        )
    elif kelly_roi > flat_roi + 0.005:
        st.info(
            "Kelly lost less than flat staking on this sample, and both are still down. "
            "Sizing did not turn the prices into an edge."
        )
    st.caption(
        f"Flat ROI {flat_roi:+.1%} on {len(bets)} units. "
        f"Kelly ROI {kelly_roi:+.1%} on the same total amount."
    )

    curve = cumulative_frame(bets)
    chart = (
        alt.Chart(curve)
        .mark_line()
        .encode(
            x=alt.X("game_date:T", title="Date"),
            y=alt.Y("Profit (u):Q", title="Cumulative profit (units)"),
            color=alt.Color(
                "Strategy:N",
                scale=alt.Scale(
                    domain=["Flat profit", "Kelly-weighted profit"],
                    range=["#f58426", "#7dce6a"],
                ),
            ),
            tooltip=["game_date:T", "Strategy:N", alt.Tooltip("Profit (u):Q", format="+.2f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")

    monthly = monthly_table(bets)
    show = monthly.copy()
    show["Hit rate"] = show["Hit rate"].map(lambda value: f"{value:.1%}")
    show["Flat profit (u)"] = show["Flat profit (u)"].map(lambda value: f"{value:+.2f}")
    show["Flat ROI"] = show["Flat ROI"].map(lambda value: f"{value:+.1%}")
    show["Kelly profit (u)"] = show["Kelly profit (u)"].map(lambda value: f"{value:+.2f}")
    show["Kelly ROI"] = show["Kelly ROI"].map(lambda value: "—" if pd.isna(value) else f"{value:+.1%}")
    st.dataframe(show, width="stretch", hide_index=True)

    bars = monthly.melt(
        id_vars=["Month", "Bets"],
        value_vars=["Flat ROI", "Kelly ROI"],
        var_name="Strategy",
        value_name="ROI",
    ).dropna()
    bar_chart = (
        alt.Chart(bars)
        .mark_bar()
        .encode(
            x=alt.X("Month:N", sort=list(monthly["Month"]), title="Month"),
            xOffset="Strategy:N",
            y=alt.Y("ROI:Q", title="ROI", axis=alt.Axis(format=".0%")),
            color=alt.Color(
                "Strategy:N",
                scale=alt.Scale(domain=["Flat ROI", "Kelly ROI"], range=["#f58426", "#7dce6a"]),
            ),
            tooltip=["Month:N", "Strategy:N", alt.Tooltip("ROI:Q", format="+.1%"), "Bets:Q"],
        )
        .properties(height=280)
    )
    st.altair_chart(bar_chart, width="stretch")
    st.dataframe(bet_table(bets, float(unit_dollars)), width="stretch", hide_index=True)


st.markdown(
    """
    <section class="nba-hero">
      <h1>NBA Moneyline</h1>
      <p>Win probabilities compared with the DraftKings line. Profit settles each season against that book’s close.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

today = date.today()
if "target_date" not in st.session_state:
    st.session_state["target_date"] = today


def choose_tomorrow() -> None:
    st.session_state["target_date"] = today + timedelta(days=1)


with st.sidebar:
    st.header("Controls")
    target_date = st.date_input(
        "Game date",
        key="target_date",
        min_value=date(2025, 10, 1),
        max_value=today + timedelta(days=7),
    )
    is_past = target_date < today
    recommendations_only = st.toggle(
        "Recommendations only",
        value=False,
        disabled=is_past,
        help="Show only sides whose margin clears the threshold.",
    )
    edge_percent = st.slider(
        "Minimum edge",
        min_value=0,
        max_value=20,
        value=5,
        step=1,
        format="%d%%",
        disabled=is_past or not recommendations_only,
        help="A recommendation appears when model probability exceeds the vig-free line by this amount.",
    )
    bankroll = st.number_input(
        "Bankroll ($)",
        min_value=0,
        value=1000,
        step=100,
        disabled=is_past,
        help="Turns each recommended stake into dollars. Set to 0 to show percents only.",
    )
    if st.button("Refresh", type="primary", width="stretch"):
        cached_games.clear()
        cached_slate.clear()
        st.rerun()
    if target_date == today:
        st.button("View tomorrow", width="stretch", on_click=choose_tomorrow)
    st.caption(
        "Past dates show model probabilities only. Today and later dates use the current "
        "DraftKings moneyline. Margin is model probability minus the vig-free price."
    )

slate_tab, profit_tab = st.tabs(["Daily slate", "Profit"])

with slate_tab:
    render_daily_slate(
        target_date=target_date,
        is_past=is_past,
        recommendations_only=recommendations_only,
        edge_threshold=edge_percent / 100.0,
        bankroll=float(bankroll),
    )

with profit_tab:
    render_profit_tab()
