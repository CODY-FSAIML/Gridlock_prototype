import json
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from catboost import CatBoostClassifier


st.set_page_config(page_title="Ravis _AI.control | Event Command", page_icon="🚦", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background: #f6f8fc; }
    [data-testid="stSidebar"] { background: #0c172b; }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #eef4ff !important;
    }
    /* Streamlit/BaseWeb select controls need an explicit dark surface; otherwise
       the selected white text can look blank on a white input field. */
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
        background-color: #172b4d !important;
        border-color: #42678f !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] input,
    [data-testid="stSidebar"] div[data-baseweb="select"] span,
    [data-testid="stSidebar"] div[data-baseweb="select"] svg {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        fill: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] div,
    [data-testid="stSidebar"] [data-baseweb="slider"] div {
        color: #eef4ff !important;
    }
    .hero { padding: 1.2rem 1.5rem; border-radius: 18px; color: white;
            background: linear-gradient(110deg, #10244d, #176b87); margin-bottom: 1rem; }
    .hero h1 { margin: 0; font-size: 2.1rem; }
    .hero p { margin: .35rem 0 0; opacity: .9; }
    .plan { padding: 1.2rem 1.35rem; border-radius: 14px; border: 1px solid #dbe4f0;
            background: white; margin: .4rem 0 .8rem; }
    .critical { border-left: 7px solid #dc2626; }
    .elevated { border-left: 7px solid #f59e0b; }
    .stable { border-left: 7px solid #10b981; }
    .eyebrow { color: #64748b; font-size: .78rem; font-weight: 700; letter-spacing: .08em; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("event_closure_model.cbm")
    return model


@st.cache_data
def load_data():
    df = pd.read_csv("gridlock_dataset.csv")
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce", utc=True)
    df["hour"] = df["start_datetime"].dt.hour.fillna(12).astype(int)
    df["is_weekend"] = df["start_datetime"].dt.dayofweek.isin([5, 6]).astype(int)
    for column in ["event_type", "event_cause", "veh_type", "corridor", "junction", "zone"]:
        df[column] = df[column].fillna("Missing").astype(str)
    df["requires_road_closure"] = df["requires_road_closure"].fillna(False).astype(bool)
    return df


@st.cache_data
def load_metrics():
    try:
        with open("model_metrics.json", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return None


try:
    model = load_model()
    historical_df = load_data()
    model_metrics = load_metrics()
except Exception as exc:
    st.error(f"Unable to load the model or dataset: {exc}")
    st.stop()


def options_for(column, fallback):
    values = sorted(v for v in historical_df[column].unique() if v != "Missing")
    return values or fallback


def build_plan(closure_risk, attendance, duration_hours, affected_lanes, peak_hour):
    """Convert model risk and event-operational inputs to a transparent starting plan."""
    attendance_factor = min(attendance / 5000, 1)
    duration_factor = min(duration_hours / 6, 1)
    lane_factor = min(affected_lanes / 3, 1)
    attendance_score = attendance_factor * 0.18
    duration_score = duration_factor * 0.10
    lane_score = lane_factor * 0.16
    peak_score = 0.12 if peak_hour else 0
    disruption_score = min(1, closure_risk * 0.58 + attendance_score + duration_score + lane_score + peak_score)

    if disruption_score >= 0.62:
        level, css = "CRITICAL", "critical"
        directive = "Heavy traffic build-up is expected. Deploy the traffic team and keep diversions ready before the event starts."
    elif disruption_score >= 0.36:
        level, css = "ELEVATED", "elevated"
        directive = "Traffic may slow down in this stretch. Keep the response team on standby and earmark one diversion route."
    else:
        level, css = "STABLE", "stable"
        directive = "Traffic impact is likely to remain manageable. Continue normal patrol and escalate if queues start building up."

    # Scale resources continuously so every operational parameter has a visible,
    # practical effect—not only when the severity band changes.
    officers = round(
        3
        + disruption_score * 9
        + attendance_factor * 4
        + duration_factor * 2
        + affected_lanes * 2
        + (2 if peak_hour else 0)
    )
    barricades = round(
        6
        + disruption_score * 17
        + attendance_factor * 6
        + duration_factor * 3
        + affected_lanes * 6
    )
    officers = int(np.clip(officers, 3, 30))
    barricades = int(np.clip(barricades, 6, 60))
    diversions = 2 if disruption_score >= 0.70 or (affected_lanes >= 3 and peak_hour) else 1 if disruption_score >= 0.38 or affected_lanes >= 2 else 0
    return {
        "score": disruption_score,
        "level": level,
        "css": css,
        "officers": officers,
        "barricades": barricades,
        "diversions": diversions,
        "resource_note": f"Calculated from {closure_risk:.0%} closure risk, {attendance:,} expected people, {duration_hours:g}-hour duration and {affected_lanes} affected lane(s).",
        "directive": directive,
    }


def obstacle_response(event_cause, affected_lanes, level, diversions):
    """Give a control-room-friendly response for the selected obstruction type."""
    cause_actions = {
        "vehicle_breakdown": "Move the vehicle to the shoulder or service lane. Call the recovery vehicle if it cannot be moved immediately.",
        "accident": "Create a safe cordon first. Keep an ambulance and recovery vehicle route clear; do not allow bystanders near the carriageway.",
        "tree_fall": "Cordon the affected stretch and request the tree-clearance team. Keep traffic away until branches and debris are removed.",
        "water_logging": "Deploy cones before the waterlogged patch and inform the pumping or civic-maintenance team. Stop vehicles from entering deep water.",
        "construction": "Coordinate with the contractor, mark the work zone clearly, and ensure machinery or material does not block the running lane.",
        "public_event": "Keep pedestrian entry and exit separate from vehicle movement. Restrict roadside parking near the venue.",
        "procession": "Keep the procession on the agreed route and deploy staff at each junction to prevent cross-traffic conflict.",
        "protest": "Secure the gathering area, maintain an emergency corridor, and keep one alternate route ready for through traffic.",
    }
    immediate = cause_actions.get(event_cause, "Secure the affected point, identify the obstruction, and clear it without blocking emergency access.")

    if affected_lanes == 0:
        arrangement = "Keep traffic moving normally. Place warning cones only if there is pedestrian or roadside activity."
    elif affected_lanes == 1:
        arrangement = "Create an early merge using cones 100–150 metres before the point. Post one officer at the merge location."
    else:
        arrangement = f"Use barricades to close or taper {affected_lanes} affected lane(s). Start the diversion before the junction so vehicles can turn safely."

    if level == "CRITICAL":
        trigger = f"Activate {diversions} diversion route(s) now. Escalate to the senior control-room officer if queues reach the previous junction."
    elif level == "ELEVATED":
        trigger = "Keep the diversion route on standby. Escalate if traffic stops moving for more than 10 minutes or queues begin spilling into the next junction."
    else:
        trigger = "Continue patrol monitoring. Escalate if the obstruction grows, pedestrian movement increases, or queue length starts rising."
    return immediate, arrangement, trigger


causes = options_for("event_cause", ["public_event", "procession", "protest", "accident"])
corridors = options_for("corridor", ["Non-corridor"])
vehicles = options_for("veh_type", ["Missing"])
junctions = options_for("junction", ["Missing"])

with st.sidebar:
    st.title("Ravis _AI.control")
    st.caption("Predictive Event Command Center")
    st.divider()
    st.subheader("Event scenario")
    event_type = st.radio("Event classification", ["planned", "unplanned"], horizontal=True)
    cause_index = causes.index("public_event") if "public_event" in causes else 0
    event_cause = st.selectbox("Event / incident type", causes, index=cause_index)
    junction = st.selectbox("Venue or nearest junction", junctions)
    corridor = st.selectbox("Affected corridor", corridors)
    unspecified_zone = "Not specified (dataset has no zone)"
    zone_choices = [unspecified_zone] + options_for("zone", ["Missing"])
    inferred_zone = historical_df.loc[historical_df["junction"] == junction, "zone"].mode()
    default_zone = inferred_zone.iloc[0] if not inferred_zone.empty and inferred_zone.iloc[0] in zone_choices else unspecified_zone
    selected_zone = st.selectbox("Traffic-management zone", zone_choices, index=zone_choices.index(default_zone))
    # Keep the UI wording friendly while retaining the category used at training time.
    zone = "Missing" if selected_zone == unspecified_zone else selected_zone
    st.divider()
    st.subheader("Operational conditions")
    event_hour = st.slider("Peak activity hour", 0, 23, 18, format="%02d:00")
    is_weekend = st.toggle("Weekend", value=True)
    attendance = st.number_input("Expected attendance / crowd", min_value=0, max_value=100000, value=2500, step=250)
    duration_hours = st.slider("Event duration (hours)", 1.0, 12.0, 3.0, 0.5)
    affected_lanes = st.slider("Lanes affected", 0, 4, 1)
    vehicle = st.selectbox("Vehicle / asset involved", ["Missing"] + vehicles)


input_df = pd.DataFrame(
    {
        "event_type": [event_type],
        "event_cause": [event_cause],
        "veh_type": [vehicle],
        "hour_of_day": [event_hour],
        "is_weekend": [int(is_weekend)],
        "corridor": [corridor],
        "zone": [zone],
        "junction": [junction],
    }
)
closure_risk = float(model.predict_proba(input_df)[0][1])
peak_hour = 7 <= event_hour <= 10 or 16 <= event_hour <= 20
plan = build_plan(closure_risk, attendance, duration_hours, affected_lanes, peak_hour)

st.markdown(
    """<div class="hero"><h1>Event Operations Plan</h1>
    <p>Forecast disruption, deploy proportionately, and capture outcomes for the next event.</p></div>""",
    unsafe_allow_html=True,
)

metric_1, metric_2, metric_3, metric_4 = st.columns(4)
metric_1.metric("Closure risk", f"{closure_risk:.0%}", help="Baseline likelihood from the historical incident model.")
metric_2.metric("Disruption severity", plan["level"], f"{plan['score']:.0%} composite")
metric_3.metric("Recommended officers", plan["officers"])
metric_4.metric("Barricades / diversion points", f"{plan['barricades']} / {plan['diversions']}")

left, right = st.columns([1.15, 1])
with left:
    st.markdown("### Recommended starting deployment")
    st.markdown(
        f"""<div class="plan {plan['css']}"><div class="eyebrow">{plan['level']} OPERATIONAL STATUS</div>
        <h3 style="margin:.35rem 0">{plan['directive']}</h3>
        <p style="margin:0">Stage <b>{plan['officers']} officers</b> and <b>{plan['barricades']} barricades</b> 
        {"with " + str(plan['diversions']) + " diversion route(s)" if plan['diversions'] else "with no diversion activated initially"}.
        Begin staging <b>{45 if plan['level'] == 'CRITICAL' else 30 if plan['level'] == 'ELEVATED' else 15} minutes</b> before peak activity.</p></div>""",
        unsafe_allow_html=True,
    )
    st.caption(plan["resource_note"])

    drivers = []
    if peak_hour:
        drivers.append("This falls during the Bengaluru peak-hour window, when traffic movement is already heavy.")
    if attendance >= 2500:
        drivers.append(f"An expected crowd of {attendance:,} may increase pedestrian crossing and pick-up/drop-off movement.")
    if affected_lanes >= 1:
        drivers.append(f"With {affected_lanes} lane(s) affected, the usable road space will reduce and vehicles may queue up.")
    if closure_risk >= 0.5:
        drivers.append("Similar incidents in the historical records have shown a higher chance of road closure or diversion.")
    if not drivers:
        drivers.append("The selected time is comparatively less busy and the road impact is limited, so a lean deployment is sufficient.")
    st.markdown("### AI reasoning: why this plan is suggested")
    for driver in drivers:
        st.write(f"• {driver}")
    st.caption("This is a suggested starting deployment. The traffic control room can increase or reduce resources based on the ground situation.")

with right:
    st.markdown("### Venue intelligence")
    venue = historical_df[(historical_df["junction"] == junction) & historical_df["latitude"].notna() & historical_df["longitude"].notna()]
    if not venue.empty:
        point = venue.groupby("junction", as_index=False).agg(latitude=("latitude", "median"), longitude=("longitude", "median"))
        st.pydeck_chart(
            pdk.Deck(
                initial_view_state=pdk.ViewState(latitude=point.latitude.iloc[0], longitude=point.longitude.iloc[0], zoom=13, pitch=0),
                layers=[pdk.Layer("ScatterplotLayer", data=point, get_position="[longitude, latitude]", get_radius=180, get_fill_color="[220, 38, 38, 190]", pickable=True)],
                tooltip={"text": "{junction}"},
            )
        )
    else:
        st.info("No mapped coordinates are available for this junction. Select another junction to see the command map.")

    similar = historical_df[
        (historical_df["event_cause"] == event_cause)
        & (historical_df["corridor"] == corridor)
    ].copy()
    similar_count = len(similar)
    similar_closure_rate = similar["requires_road_closure"].mean() if similar_count else np.nan
    st.metric("Comparable historical records", similar_count)
    if similar_count:
        st.caption(f"{similar_closure_rate:.0%} of comparable records required a road closure.")
    else:
        st.caption("No exact match found; the model uses broader historical patterns.")

st.markdown("### On-ground obstacle response plan")
st.caption("A quick operational checklist for the traffic team at the selected location.")
immediate_action, traffic_arrangement, escalation_trigger = obstacle_response(
    event_cause, affected_lanes, plan["level"], plan["diversions"]
)
response_1, response_2, response_3 = st.columns(3)
with response_1:
    st.markdown("#### 1. First action")
    st.info(immediate_action, icon="🚧")
with response_2:
    st.markdown("#### 2. Traffic arrangement")
    st.info(traffic_arrangement, icon="🚦")
with response_3:
    st.markdown("#### 3. Escalation trigger")
    st.info(escalation_trigger, icon="📞")

st.divider()
with st.expander("Close the learning loop: record post-event outcome"):
    st.write("Saving this result makes the next planning cycle auditable. This demo stores feedback only for the active session.")
    actual_level = st.select_slider("Actual observed severity", options=["Stable", "Elevated", "Critical"], value="Elevated")
    actual_officers = st.number_input("Officers actually deployed", min_value=0, value=int(plan["officers"]))
    notes = st.text_area("What happened?", placeholder="Queue spillback, diversion effectiveness, clearance time…")
    if st.button("Save post-event feedback"):
        record = {"scenario": f"{event_cause} at {junction}", "forecast": plan["level"], "actual": actual_level, "officers": actual_officers, "notes": notes}
        st.session_state.setdefault("feedback", []).append(record)
        st.success("Outcome captured for this session. In production, this feeds model monitoring and retraining.")

if st.session_state.get("feedback"):
    st.caption(f"Session learning records: {len(st.session_state['feedback'])}")

if model_metrics:
    with st.expander("Model validation (future-period holdout)"):
        st.caption("The model is evaluated on the latest 15% of events, after training only on earlier records. F1, precision, recall and PR-AUC are the primary metrics for this imbalanced classification task.")
        evaluation_columns = st.columns(5)
        evaluation_columns[0].metric("F1", model_metrics["f1"])
        evaluation_columns[1].metric("Precision", model_metrics["precision"])
        evaluation_columns[2].metric("Recall", model_metrics["recall"])
        evaluation_columns[3].metric("PR-AUC", model_metrics["pr_auc"])
        evaluation_columns[4].metric("ROC-AUC", model_metrics["roc_auc"])
        st.caption(f"F1 decision threshold: {model_metrics['selected_threshold']:.2f} · Brier score: {model_metrics['brier_score']} · Test rows: {model_metrics['test_rows']}")

st.caption("Ravis _AI.control • Historical incident baseline + transparent operational planning • Recommendations support, not replace, traffic-control decisions.")
