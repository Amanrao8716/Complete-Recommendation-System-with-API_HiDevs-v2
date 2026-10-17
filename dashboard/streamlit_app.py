"""Streamlit dashboard: try the recommendation system in a browser.

Run from the project root::

    pip install -r requirements-ui.txt
    streamlit run dashboard/streamlit_app.py

By default it runs the engine *inside* Streamlit (no server needed). Choose
"Connect to a running API" in the sidebar to use a server started with
``python -m api.app`` instead.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any, TypeVar

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard.api_client import (  # noqa: E402
    DEFAULT_API_URL,
    FEEDBACK_TYPES,
    STRATEGIES,
    ApiClientError,
    HttpClient,
    InProcessClient,
    RecsysClient,
    build_inprocess_client,
)

T = TypeVar("T")
BUILTIN = "Built-in demo (no server needed)"
REMOTE = "Connect to a running API"


@st.cache_resource(show_spinner="Preparing the demo database...")
def _builtin_client() -> InProcessClient:
    """Create the in-process app once and reuse it across reruns."""
    return build_inprocess_client()


def _attempt(action: Callable[[], T]) -> T | None:
    """Run an API call; on failure show the error and return ``None``."""
    try:
        return action()
    except ApiClientError as exc:
        st.error(str(exc))
        return None


def _sidebar() -> RecsysClient:
    """Draw the sidebar and return the client chosen there."""
    st.sidebar.title("Recommendation System")
    mode = st.sidebar.radio("Data source", [BUILTIN, REMOTE], key="mode")
    client: RecsysClient
    if mode == BUILTIN:
        try:
            client = _builtin_client()
        except Exception as exc:  # noqa: BLE001 - show any start-up problem
            st.sidebar.error(f"Could not start the demo: {exc}")
            st.stop()
    else:
        url = st.sidebar.text_input(
            "API URL",
            os.environ.get("RECSYS_API_URL", DEFAULT_API_URL),
            key="api_url",
        )
        key = st.sidebar.text_input(
            "API key (optional)", type="password", key="api_key"
        )
        client = HttpClient(url, key)
    try:
        health = client.health()
        st.sidebar.success(
            f"API {health['status']}: {health.get('users', '?')} users, "
            f"{health.get('content_items', '?')} content items"
        )
    except ApiClientError as exc:
        st.sidebar.error(str(exc))
    st.sidebar.caption("Endpoints: /recommend, /feedback, /health, /metrics")
    return client


def _render_recommendations(data: dict[str, Any]) -> None:
    """Show one ``/recommend`` response as ranked cards."""
    freshness = "from cache" if data["cached"] else "freshly computed"
    st.caption(
        f"Strategy **{data['strategy']}** · {data['count']} results · "
        f"{freshness} · {data['latency_ms']} ms · "
        f"request `{data['request_id']}`"
    )
    rows = data["recommendations"]
    if not rows:
        st.info("No recommendations for this user.")
        return
    for rank, rec in enumerate(rows, start=1):
        with st.container(border=True):
            left, right = st.columns([4, 1])
            left.subheader(f"{rank}. {rec['title']}")
            left.caption(
                f"{rec['category']} · difficulty {rec['difficulty']}/5 · "
                f"content ID {rec['content_id']}"
            )
            right.metric("Score", f"{rec['score']:.2f}")
            st.progress(min(1.0, max(0.0, float(rec["score"]))))
            st.write(rec["explanation"])
            with st.expander("Why this score? (component breakdown)"):
                parts = pd.DataFrame(
                    {
                        "component": list(rec["components"]),
                        "score": list(rec["components"].values()),
                    }
                ).set_index("component")
                st.bar_chart(parts)


def _tab_recommendations(client: RecsysClient) -> None:
    """Ask for recommendations and show them."""
    c1, c2, c3 = st.columns(3)
    user_id = int(c1.number_input("User ID", 1, value=1, step=1, key="r_user"))
    limit = int(c2.slider("How many", 1, 20, 5, key="r_limit"))
    strategy = c3.selectbox("Strategy", STRATEGIES, key="r_strategy")
    st.caption(
        "Users 1-40 have history; try 41+ for a new (cold-start) user in "
        "the built-in demo."
    )
    if st.button("Get recommendations", type="primary", key="r_go"):
        st.session_state["recs"] = _attempt(
            lambda: client.recommend(user_id, limit, strategy)
        )
    data = st.session_state.get("recs")
    if data:
        _render_recommendations(data)


def _ranking_table(data: dict[str, Any]) -> pd.DataFrame:
    """Compact rank / title / score table for before-and-after views."""
    return pd.DataFrame(
        [
            {
                "rank": i,
                "title": r["title"],
                "score": r["score"],
            }
            for i, r in enumerate(data["recommendations"], start=1)
        ]
    )


def _tab_feedback(client: RecsysClient) -> None:
    """Send feedback and compare recommendations before and after."""
    st.markdown(
        "Send an interaction, then compare the user's recommendations "
        "**before** and **after**. Feedback is stored and the user's cache "
        "is invalidated immediately."
    )
    c1, c2, c3 = st.columns(3)
    user_id = int(c1.number_input("User ID", 1, value=1, step=1, key="f_user"))
    content_id = int(
        c2.number_input("Content ID", 1, value=7, step=1, key="f_content")
    )
    kind = c3.selectbox("Interaction", FEEDBACK_TYPES, index=2, key="f_type")
    rating = None
    if kind == "rate":
        rating = int(st.slider("Rating", 1, 5, 5, key="f_rating"))
    try:
        current = client.recommend(user_id, 10)
        ids = [r["content_id"] for r in current["recommendations"]]
        st.caption(f"Tip: recommended content IDs for this user: {ids}")
    except ApiClientError:
        pass  # the real error is shown when the button is pressed
    if not st.button("Send feedback", type="primary", key="f_send"):
        return
    before = _attempt(lambda: client.recommend(user_id, 5, "auto"))
    if before is None:
        return
    stored = _attempt(
        lambda: client.feedback(user_id, content_id, kind, rating)
    )
    if stored is None:
        return
    after = _attempt(lambda: client.recommend(user_id, 5, "auto"))
    st.success(
        f"Recorded '{kind}' on content {content_id} for user {user_id} "
        f"(interaction {stored['interaction']['interaction_id']})."
    )
    if after is None:
        return
    left, right = st.columns(2)
    left.subheader("Before")
    left.dataframe(_ranking_table(before), hide_index=True)
    right.subheader("After")
    right.dataframe(_ranking_table(after), hide_index=True)
    if before["recommendations"] == after["recommendations"]:
        st.info(
            "The list did not change. Try 'complete' on an item related to "
            "the user's interests, or give several interactions."
        )


def _tab_system(client: RecsysClient) -> None:
    """Show health, request metrics and cache statistics."""
    st.button("Refresh", key="s_refresh")  # any click reruns the page
    health = _attempt(client.health)
    if health:
        a, b, c = st.columns(3)
        a.metric("Status", health["status"])
        b.metric("Users", health.get("users", "?"))
        c.metric("Content items", health.get("content_items", "?"))
    metrics = _attempt(client.metrics)
    if not metrics:
        return
    cache = metrics["recommender"]["cache"]
    d, e, f = st.columns(3)
    d.metric("Total requests", metrics["total_requests"])
    e.metric("Uptime (s)", metrics["uptime_seconds"])
    f.metric("Cache hit rate", f"{cache['hit_rate']:.0%}")
    if metrics["endpoints"]:
        st.subheader("Latency per endpoint (ms)")
        st.dataframe(pd.DataFrame(metrics["endpoints"]).T)
    if metrics["status_codes"]:
        st.subheader("Responses by status code")
        st.bar_chart(pd.Series(metrics["status_codes"], name="requests"))


def main() -> None:
    """Build the page."""
    st.set_page_config(
        page_title="Recommendation System", page_icon="🎯", layout="wide"
    )
    client = _sidebar()
    st.title("🎯 Learning-Content Recommendation System")
    st.caption(
        "Hybrid recommendations with explanations, a feedback loop, caching "
        "and live metrics."
    )
    tab_rec, tab_fb, tab_sys = st.tabs(
        ["Recommendations", "Give feedback", "System"]
    )
    with tab_rec:
        _tab_recommendations(client)
    with tab_fb:
        _tab_feedback(client)
    with tab_sys:
        _tab_system(client)


main()
