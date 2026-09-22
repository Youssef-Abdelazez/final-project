"""Streamlit student and educator views for the Phase 8 demonstrator."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from phase8.app.api_client import APIClient, APIClientError
from phase8.app.config import get_settings


SOURCE_LABELS = {
    "lstm_dkt_early_history": "Personal practice history",
    "discovery_skill_prior": "General learner pattern",
    "unavailable": "Evidence unavailable",
}

ROUTE_LABELS = {
    "cf": "Similar learners",
    "sequential": "Recent learning order",
    "content": "Related skills",
    "popularity": "Overall popularity",
    "dkt": "Estimated learning level",
    "unresolved_base": "Available learning information",
}


def source_label(source: str | None) -> str:
    return SOURCE_LABELS.get(source or "unavailable", "Other saved evidence")


def format_percent(value: Any) -> str:
    if value is None:
        return "Unavailable"
    return f"{float(value):.0%}"


def numeric_frame(rows: list[dict[str, Any]], label: str, value: str) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty or label not in frame or value not in frame:
        return pd.DataFrame(columns=[label, value])
    frame[value] = pd.to_numeric(frame[value], errors="coerce").fillna(0)
    return frame[[label, value]]


def saved_metric_rows(overview: dict[str, Any], source: str) -> list[dict[str, Any]]:
    return [
        row["values"] for row in overview.get("saved_metrics", [])
        if row.get("source") == source
    ]


def render_prototype_notice() -> None:
    st.warning(
        "Research prototype: these are model estimates from historical data."
    )


def render_profile_summary(profile: dict[str, Any]) -> None:
    columns = st.columns(3)
    columns[0].metric("Early interactions used", profile.get("supported_early_interactions") or 0)
    columns[1].metric("Evidence source", source_label(profile.get("state_source")))
    columns[2].metric(
        "Problem-history status",
        "No supported history" if profile.get("cold_start_problem_history") else "History available",
    )
    if profile.get("state_source") == "discovery_skill_prior":
        st.info(
            "There was not enough supported personal history for a recurrent estimate. "
            "The displayed learning-state evidence therefore uses general patterns learned "
            "from the discovery cohort."
        )


def render_skill_chart(skills: list[dict[str, Any]]) -> None:
    st.subheader("Estimated skill state")
    st.caption(
        "Model-estimated probability of a correct response. This is not a formal grade or diagnosis."
    )
    if not skills:
        st.info("No supported skill-state estimates are available for this learner.")
        return
    frame = pd.DataFrame(skills)
    frame["predicted_correctness"] = pd.to_numeric(
        frame["predicted_correctness"], errors="coerce"
    )
    frame = frame.dropna(subset=["predicted_correctness"]).nsmallest(
        12, "predicted_correctness"
    )
    chart = frame.set_index("skill_id")[["predicted_correctness"]]
    st.bar_chart(chart, horizontal=True, x_label="Estimated correctness", y_label="Skill")
    with st.expander("View chart values"):
        values = frame[["skill_id", "predicted_correctness"]].copy()
        values["predicted_correctness"] = values["predicted_correctness"].map(format_percent)
        st.dataframe(values, hide_index=True, use_container_width=True)


def render_recommendations(recommendations: list[dict[str, Any]]) -> None:
    st.subheader("Recommended practice")
    st.caption(
        "The order is fixed from the final offline model. Explanations describe recorded ranking evidence"
    )
    if not recommendations:
        st.info("No recommendations are available for this learner and application slice.")
        return
    for recommendation in recommendations:
        title = recommendation.get("problem_label") or recommendation["item_id"]
        st.markdown(f"**{recommendation['rank']}. {title}**")
        st.write(recommendation["dashboard_explanation"])
        warnings = recommendation.get("warning_messages", [])
        if warnings:
            st.caption("Evidence limitation: " + " ".join(warnings))
        with st.expander("Why this item?"):
            skill = recommendation.get("skill_label") or recommendation.get("primary_skill_id")
            st.write(f"Supporting skill: {skill or 'No supported skill label is available.'}")
            st.write(
                "Estimated correctness: "
                + format_percent(recommendation.get("predicted_correctness"))
            )
            st.write("Evidence source: " + source_label(recommendation.get("state_source")))
            st.write("Main evidence type: " + recommendation.get("route_label", "Available evidence"))
            for warning in warnings:
                st.warning(warning)
        st.divider()


def render_learner_view(client: APIClient, learner_id: str, *, heading: str) -> None:
    st.header(heading)
    profile = client.profile(learner_id)
    skills = client.skills(learner_id)
    recommendations = client.recommendations(learner_id)
    st.caption(f"Pseudonymous learner: {learner_id}")
    render_profile_summary(profile)
    render_skill_chart(skills)
    render_recommendations(recommendations)


def choose_learner(client: APIClient, *, key_prefix: str) -> str | None:
    query = st.text_input(
        "Search pseudonymous learner ID", key=f"{key_prefix}_search",
        help="Only pseudonymous identifiers are available. Names and raw responses are not served.",
    )
    learners = client.learners(query=query or None, limit=100)
    if not learners:
        st.info("No learner matches this search.")
        return None
    return st.selectbox(
        "Select learner", [row["learner_id"] for row in learners],
        key=f"{key_prefix}_learner",
    )


def render_student_view(client: APIClient) -> None:
    st.title("My practice recommendations")
    render_prototype_notice()
    learner_id = choose_learner(client, key_prefix="student")
    if learner_id:
        render_learner_view(client, learner_id, heading="Learning evidence")


def render_distribution(
    title: str, frame: pd.DataFrame, *, label: str, value: str, explanation: str
) -> None:
    st.subheader(title)
    st.caption(explanation)
    if frame.empty:
        st.info("No saved values are available for this summary.")
        return
    st.bar_chart(frame.set_index(label)[[value]], horizontal=True)
    with st.expander(f"View {title.lower()} values"):
        st.dataframe(frame, hide_index=True, use_container_width=True)


def render_evaluation_summary(overview: dict[str, Any]) -> None:
    st.subheader("Offline evaluation context")
    st.info(
        "The selected sequential hybrid produced a small positive point improvement over the weighted hybrid."
    )
    decision = saved_metric_rows(overview, "phase6_decision.csv")
    uncertainty = saved_metric_rows(overview, "paired_uncertainty.csv")
    with st.expander("View saved Phase 6 evaluation rows"):
        if decision:
            st.markdown("**Final model decision**")
            st.dataframe(pd.DataFrame(decision), hide_index=True, use_container_width=True)
        if uncertainty:
            st.markdown("**Paired uncertainty estimates**")
            st.dataframe(pd.DataFrame(uncertainty), hide_index=True, use_container_width=True)
        if not decision and not uncertainty:
            st.caption("Phase 6 summary rows are not present in this database build.")


def render_educator_view(client: APIClient) -> None:
    st.title("Educator research overview")
    render_prototype_notice()
    st.caption(
        "This is an aggregate research-cohort view. Indicators support human review."
    )
    overview = client.teacher_overview()
    summary_columns = st.columns(3)
    summary_columns[0].metric("Pseudonymous learners", overview.get("learner_count", 0))
    summary_columns[1].metric("Displayed recommendations", overview.get("recommendation_count", 0))
    summary_columns[2].metric(
        "Recommendation limit",
        "10 per learner",
    )

    route_rows = overview.get("routes", [])
    route_frame = numeric_frame(route_rows, "route_label", "recommendation_count")
    render_distribution(
        "Recommendation evidence routes", route_frame,
        label="route_label", value="recommendation_count",
        explanation="Counts use the saved primary recommendation slice.",
    )

    evidence_rows = [
        {"Evidence source": source_label(row.get("state_source")), "Learners": row.get("learner_count", 0)}
        for row in overview.get("evidence_sources", [])
    ]
    evidence_frame = numeric_frame(evidence_rows, "Evidence source", "Learners")
    render_distribution(
        "Learner-state evidence", evidence_frame,
        label="Evidence source", value="Learners",
        explanation="Personal practice history is separated from general discovery-cohort fallback evidence.",
    )

    history_rows = [{
        "Problem-history group": (
            "No supported history" if row.get("cold_start_problem_history") else "History available"
        ),
        "Learners": row.get("learner_count", 0),
    } for row in overview.get("history_groups", [])]
    history_frame = numeric_frame(history_rows, "Problem-history group", "Learners")
    render_distribution(
        "Problem-history subgroups", history_frame,
        label="Problem-history group", value="Learners",
        explanation="This is a descriptive evidence-availability split.",
    )

    warning_rows = [
        {"Limitation": row["message"], "Recommendations": row["recommendation_count"]}
        for row in overview.get("warnings", []) if row.get("code") != "none"
    ]
    warning_frame = numeric_frame(warning_rows, "Limitation", "Recommendations")
    render_distribution(
        "Explanation limitations", warning_frame,
        label="Limitation", value="Recommendations",
        explanation="One recommendation can carry more than one transparent evidence limitation.",
    )
    render_evaluation_summary(overview)

    st.header("Review learner evidence")
    cold_choice = st.selectbox(
        "Problem-history status", ["All", "History available", "No supported history"]
    )
    state_options = ["All"] + [
        row["state_source"] for row in overview.get("evidence_sources", [])
        if row.get("state_source") != "unavailable"
    ]
    state_choice = st.selectbox(
        "Learner-state evidence", state_options,
        format_func=lambda value: value if value == "All" else source_label(value),
    )
    route_options = ["All"] + [row["route"] for row in route_rows if row.get("route")]
    route_choice = st.selectbox(
        "Recommendation evidence route", route_options,
        format_func=lambda value: value if value == "All" else ROUTE_LABELS.get(value, value),
    )
    warning_options = ["All"] + [
        row["code"] for row in overview.get("warnings", []) if row.get("code") != "none"
    ]
    warning_lookup = {
        row["code"]: row["message"] for row in overview.get("warnings", [])
    }
    warning_choice = st.selectbox(
        "Explanation limitation", warning_options,
        format_func=lambda value: value if value == "All" else warning_lookup.get(value, value),
    )
    cold_start = {
        "All": None, "History available": False, "No supported history": True,
    }[cold_choice]
    learners = client.teacher_learners(
        cold_start=cold_start,
        state_source=None if state_choice == "All" else state_choice,
        dominant_route=None if route_choice == "All" else route_choice,
        warning=None if warning_choice == "All" else warning_choice,
        limit=200,
    )
    if not learners:
        st.info("No learners match these evidence filters.")
        return
    display_rows = [{
        "Learner": row["learner_id"],
        "History": "No supported history" if row["cold_start_problem_history"] else "Available",
        "Evidence source": source_label(row.get("state_source")),
        "Early interactions": row.get("supported_early_interactions"),
    } for row in learners]
    st.dataframe(pd.DataFrame(display_rows), hide_index=True, use_container_width=True)
    learner_id = st.selectbox(
        "Open learner evidence", [row["learner_id"] for row in learners],
        key="educator_drilldown",
    )
    render_learner_view(client, learner_id, heading="Learner evidence details")


def render_glossary() -> None:
    with st.sidebar.expander("Evidence glossary"):
        st.markdown(
            "- **Similar learners:** patterns from learners with comparable histories.\n"
            "- **Related skills:** overlap with skills in recent work.\n"
            "- **Recent learning order:** what commonly followed recent practice.\n"
            "- **Overall popularity:** a transparent fallback when stronger evidence is unavailable.\n"
            "- **Estimated learning level:** a small adjustment from the saved learner-state model."
        )


def main() -> None:
    st.set_page_config(
        page_title="Educational Recommendation Demonstrator",
        page_icon="📘",
        layout="wide",
    )
    settings = get_settings()
    client = APIClient(settings.api_url)
    st.sidebar.title("Recommendation demonstrator")
    role = st.sidebar.radio("View", ["Student", "Educator"])
    render_glossary()
    try:
        health = client.health()
        if not health.get("database_ready"):
            st.error(
                "The API is running, but the frozen recommendation database is not ready. "
                "Build it with `python -m phase8.app.import_data`."
            )
            st.stop()
        if role == "Student":
            render_student_view(client)
        else:
            render_educator_view(client)
    except APIClientError as error:
        st.error(str(error))
        st.info(
            "Start the service with `python -m uvicorn phase8.app.api:app --reload`, "
            "then refresh this page."
        )


if __name__ == "__main__":
    main()
