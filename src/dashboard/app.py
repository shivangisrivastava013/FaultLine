"""
FaultLine Incident & Operations Dashboard (Streamlit).

Features:
- Service Health Overview & KPI Metrics
- Open & Historic Incident Inspector
- Distribution Shift Visualizer (Reference vs Incoming Data)
- Grounded AI Root-Cause Investigation Viewer with Evidence Citations
- Audited Model Rollback & Version Activation Control Panel
- Traffic Simulation & Chaos Injection Controls
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from typing import Dict, Any, List

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src.incidents.database import SessionLocal, init_db
from src.incidents.models import (
    Incident,
    PredictionRecord,
    OutcomeRecord,
    MonitoringWindow,
    DeploymentEvent,
    IncidentAction,
)
from src.incidents.manager import IncidentManager
from src.investigation.evidence import EvidenceBundleCompiler
from src.investigation.analyzer import InvestigationAnalyzer


API_BASE_URL = os.getenv("PREDICTION_API_URL", "http://localhost:8000")
REF_DATA_PATH = "data/reference_baseline.parquet"

st.set_page_config(
    page_title="FaultLine | ML Incident Observability & Recovery",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .badge-open {
        background-color: #ff4d4f;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-ack {
        background-color: #faad14;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-res {
        background-color: #52c41a;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def get_api_health() -> Dict[str, Any]:
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=2.0)
        return r.json() if r.status_code == 200 else {"status": "unhealthy"}
    except Exception:
        return {"status": "unreachable"}


def get_model_info() -> Dict[str, Any]:
    try:
        r = requests.get(f"{API_BASE_URL}/model-info", timeout=2.0)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def main():
    st.title("⚡ FaultLine: ML Incident Observability & Recovery")
    st.caption("Delivery Duration Prediction Service • Simulated Operational Environment")

    init_db()
    db = SessionLocal()

    # Sidebar: Controls & Navigation
    st.sidebar.header("🕹️ Simulation & Controls")

    # API Status in Sidebar
    health_info = get_api_health()
    model_info = get_model_info()

    if health_info.get("status") == "ok":
        st.sidebar.success(f"API Online • Active: **{model_info.get('model_version', 'v1')}**")
    else:
        st.sidebar.error(f"API {health_info.get('status').title()} at {API_BASE_URL}")

    # Chaos Injection Quick Buttons
    st.sidebar.subheader("Inject Failures (Chaos)")
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        if st.button("⏱️ Latency Spike", use_container_width=True):
            try:
                requests.post(f"{API_BASE_URL}/admin/inject-fault", json={"fault_type": "latency_spike", "delay_seconds": 0.8, "probability": 0.9})
                st.sidebar.warning("Latency delay injected!")
            except Exception as e:
                st.sidebar.error(str(e))

        if st.button("💥 Server Errors", use_container_width=True):
            try:
                requests.post(f"{API_BASE_URL}/admin/inject-fault", json={"fault_type": "server_error", "probability": 0.5, "error_status_code": 500})
                st.sidebar.warning("Server 500 error injected!")
            except Exception as e:
                st.sidebar.error(str(e))

    with col_s2:
        if st.button("📉 Degraded Model", use_container_width=True):
            try:
                requests.post(f"{API_BASE_URL}/admin/activate-model", params={"version": "candidate_degraded_v2", "actor": "release_bot", "reason": "Canary rollout of v2"})
                st.sidebar.warning("Candidate v2_degraded activated!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(str(e))

        if st.button("🔄 Reset Clean", use_container_width=True):
            try:
                requests.post(f"{API_BASE_URL}/admin/inject-fault", json={"fault_type": "none"})
                requests.post(f"{API_BASE_URL}/rollback", json={"target_version": "champion_v1", "actor": "sre", "reason": "Reset to clean"})
                st.sidebar.success("Reset to clean state!")
                st.rerun()
            except Exception as e:
                st.sidebar.error(str(e))

    # Traffic Replay trigger from UI
    st.sidebar.subheader("Traffic Replay")
    replay_count = st.sidebar.slider("Requests to replay", 10, 200, 50, 10)
    corrupt_choice = st.sidebar.selectbox("Corruption mode", ["None", "invalid_inputs", "data_drift"])
    if st.sidebar.button("🚀 Replay Traffic Now"):
        from simulation.replay_traffic import TrafficReplayer
        replayer = TrafficReplayer(api_base_url=API_BASE_URL)
        c_mode = None if corrupt_choice == "None" else corrupt_choice
        summary = replayer.replay(num_requests=replay_count, rate_per_second=20.0, corrupt_mode=c_mode, corrupt_probability=0.4)
        st.sidebar.info(f"Replayed {summary['requests_sent']} requests ({summary['success_count']} success, {summary['validation_error_count']} 422s)")
        # Trigger worker once
        from src.monitoring.worker import MonitoringWorker
        worker = MonitoringWorker()
        worker.process_window(db)
        st.rerun()

    # Main Dashboard Tabs
    tab_overview, tab_incidents, tab_drift, tab_rollback, tab_logs = st.tabs([
        "📊 Service Overview",
        "🚨 Incidents & AI Investigation",
        "📈 Feature Drift & Data Quality",
        "🛡️ Model Rollback & Audit",
        "📋 Live Request Logs",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: Service Overview
    # -------------------------------------------------------------------------
    with tab_overview:
        # Top KPI Tiles
        total_reqs = db.query(PredictionRecord).count()
        total_outcomes = db.query(OutcomeRecord).count()
        open_incidents_count = db.query(Incident).filter(Incident.status == "Open").count()
        recent_window = db.query(MonitoringWindow).order_by(MonitoringWindow.end_time.desc()).first()

        kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5 = st.columns(5)
        with kpi_col1:
            st.metric("Total Predictions", f"{total_reqs:,}")
        with kpi_col2:
            st.metric("Delivered Outcomes", f"{total_outcomes:,}")
        with kpi_col3:
            p95 = f"{recent_window.p95_latency_ms:.1f} ms" if recent_window else "N/A"
            st.metric("p95 Latency", p95)
        with kpi_col4:
            mae = f"{recent_window.mae:.2f} min" if (recent_window and recent_window.mae is not None) else "N/A"
            st.metric("Window MAE", mae)
        with kpi_col5:
            st.metric("Active Open Incidents", open_incidents_count, delta_color="inverse")

        st.divider()

        # Operational Latency & Error Charts
        recent_windows = db.query(MonitoringWindow).order_by(MonitoringWindow.end_time.desc()).limit(30).all()
        if recent_windows:
            df_w = pd.DataFrame([
                {
                    "Time": w.end_time,
                    "p95 Latency (ms)": w.p95_latency_ms,
                    "Error Rate (%)": w.error_rate,
                    "Validation Failures (%)": w.validation_failure_rate,
                    "MAE (min)": w.mae,
                }
                for w in reversed(recent_windows)
            ])

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                fig_lat = px.line(df_w, x="Time", y="p95 Latency (ms)", title="p95 Request Latency Trend")
                fig_lat.add_hline(y=150.0, line_dash="dash", line_color="orange", annotation_text="Warning (150ms)")
                fig_lat.add_hline(y=400.0, line_dash="dash", line_color="red", annotation_text="Critical (400ms)")
                st.plotly_chart(fig_lat, use_container_width=True)

            with col_c2:
                fig_err = px.line(df_w, x="Time", y=["Error Rate (%)", "Validation Failures (%)"], title="Error & Validation Failure Rates")
                fig_err.add_hline(y=5.0, line_dash="dash", line_color="red", annotation_text="5xx Error Threshold")
                st.plotly_chart(fig_err, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 2: Incidents & Grounded AI Investigation
    # -------------------------------------------------------------------------
    with tab_incidents:
        st.subheader("🚨 Incident Management & Root-Cause Investigation")

        status_filter = st.radio("Filter Status:", ["All", "Open", "Acknowledged", "Resolved"], horizontal=True)
        query = db.query(Incident)
        if status_filter != "All":
            query = query.filter(Incident.status == status_filter)
        incidents = query.order_by(Incident.last_observed_at.desc()).all()

        if not incidents:
            st.info("No incidents found matching filter. System operating smoothly!")
        else:
            for inc in incidents:
                status_color = "🔴" if inc.status == "Open" else "🟡" if inc.status == "Acknowledged" else "🟢"
                with st.expander(f"{status_color} [{inc.status.upper()}] {inc.incident_type} — ID: {inc.incident_id} ({inc.severity.upper()})"):
                    st.write(f"**Model Version Affected:** `{inc.model_version}`")
                    st.write(f"**First Detected:** `{inc.first_detected_at.isoformat()}` • **Last Observed:** `{inc.last_observed_at.isoformat()}`")
                    st.write(f"**Recommended Recovery Action:** {inc.recommended_action}")

                    # State machine actions
                    act_col1, act_col2, act_col3 = st.columns([1, 1, 3])
                    with act_col1:
                        if inc.status == "Open":
                            if st.button("Acknowledge", key=f"ack_{inc.incident_id}"):
                                IncidentManager.acknowledge_incident(db, inc.incident_id, actor="operator")
                                st.rerun()
                    with act_col2:
                        if inc.status in ["Open", "Acknowledged"]:
                            if st.button("Mark Resolved", key=f"res_{inc.incident_id}"):
                                IncidentManager.resolve_incident(db, inc.incident_id, actor="operator", resolution_notes="Resolved via dashboard")
                                st.rerun()

                    st.divider()

                    # Grounded AI Investigation Section
                    st.markdown("### 🤖 Grounded AI Root-Cause Investigation")
                    if st.button(f"Generate / Refresh Investigation for {inc.incident_id}", key=f"btn_ai_{inc.incident_id}"):
                        bundle = EvidenceBundleCompiler.compile_bundle(db, inc)
                        analyzer = InvestigationAnalyzer()
                        report = analyzer.investigate(bundle)
                        st.session_state[f"report_{inc.incident_id}"] = report

                    if f"report_{inc.incident_id}" in st.session_state:
                        rep = st.session_state[f"report_{inc.incident_id}"]
                        st.info(f"**Summary:** {rep.get('summary')}")

                        col_obs, col_cause = st.columns(2)
                        with col_obs:
                            st.markdown("#### 🔍 Grounded Observations (with Citations)")
                            for obs in rep.get("observations", []):
                                st.markdown(f"- {obs}")
                        with col_cause:
                            st.markdown("#### 💡 Hypothesized Root Causes")
                            for cause in rep.get("possible_causes", []):
                                st.markdown(f"- {cause}")

                        st.markdown("#### 📋 Recommended Verification Checks")
                        for chk in rep.get("recommended_checks", []):
                            st.markdown(f"- {chk}")

                        st.success(f"**Suggested Recovery Action:** {rep.get('suggested_action')}")

                    # Raw Evidence Drilldown
                    with st.expander("Inspect Raw Evidence JSON"):
                        st.json(inc.evidence)

    # -------------------------------------------------------------------------
    # TAB 3: Feature Drift & Data Quality
    # -------------------------------------------------------------------------
    with tab_drift:
        st.subheader("📈 Statistical Drift & Data Quality Diagnostics")

        if not os.path.exists(REF_DATA_PATH):
            st.warning("Reference baseline dataset not found at data/reference_baseline.parquet.")
        else:
            ref_df = pd.read_parquet(REF_DATA_PATH)
            # Fetch recent requests
            recent_reqs = (
                db.query(PredictionRecord)
                .filter(PredictionRecord.status_code == 200)
                .order_by(PredictionRecord.timestamp.desc())
                .limit(200)
                .all()
            )

            if not recent_reqs:
                st.info("No recent prediction requests available. Replay traffic using the sidebar controls.")
            else:
                curr_df = pd.DataFrame([
                    {
                        "distance_km": r.distance_km,
                        "traffic_level": r.traffic_level,
                        "weather": r.weather,
                        "restaurant_load": r.restaurant_load,
                        "hour_of_day": r.hour_of_day,
                        "predicted_minutes": r.predicted_minutes,
                    }
                    for r in recent_reqs
                ])

                st.write(f"Comparing **{len(curr_df)} recent incoming requests** against **{len(ref_df)} reference baseline samples**.")

                col_f1, col_f2 = st.columns(2)
                with col_f1:
                    # Distance distribution comparison
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(x=ref_df["distance_km"], name="Reference Baseline", opacity=0.6, histnorm="probability"))
                    fig_dist.add_trace(go.Histogram(x=curr_df["distance_km"], name="Recent Incoming", opacity=0.6, histnorm="probability"))
                    fig_dist.update_layout(barmode="overlay", title="Distance Distribution (distance_km)")
                    st.plotly_chart(fig_dist, use_container_width=True)

                with col_f2:
                    # Weather distribution comparison
                    ref_weather = ref_df["weather"].value_counts(normalize=True).reset_index()
                    ref_weather.columns = ["weather", "Reference"]
                    curr_weather = curr_df["weather"].value_counts(normalize=True).reset_index()
                    curr_weather.columns = ["weather", "Incoming"]
                    m_weather = pd.merge(ref_weather, curr_weather, on="weather", how="outer").fillna(0)

                    fig_w = go.Figure(data=[
                        go.Bar(name="Reference", x=m_weather["weather"], y=m_weather["Reference"]),
                        go.Bar(name="Incoming", x=m_weather["weather"], y=m_weather["Incoming"]),
                    ])
                    fig_w.update_layout(barmode="group", title="Weather Category Proportions")
                    st.plotly_chart(fig_w, use_container_width=True)

                # Predictions distribution comparison
                fig_pred = go.Figure()
                if "delivery_duration_minutes" in ref_df:
                    fig_pred.add_trace(go.Histogram(x=ref_df["delivery_duration_minutes"], name="Reference Target Duration", opacity=0.6, histnorm="probability"))
                fig_pred.add_trace(go.Histogram(x=curr_df["predicted_minutes"], name="Incoming Predictions", opacity=0.6, histnorm="probability"))
                fig_pred.update_layout(barmode="overlay", title="Prediction Duration Distribution")
                st.plotly_chart(fig_pred, use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 4: Model Rollback & Deployment Audit
    # -------------------------------------------------------------------------
    with tab_rollback:
        st.subheader("🛡️ Audited Model Rollback & Version Control")

        active_ver = model_info.get("model_version", "v1")
        st.write(f"**Currently Active Serving Model:** `{active_ver}`")

        col_rb1, col_rb2 = st.columns([1, 2])
        with col_rb1:
            st.markdown("#### Execute Rollback")
            target_v = st.selectbox("Target Model Version", ["champion_v1", "baseline_linear", "candidate_degraded_v2"])
            operator_name = st.text_input("Operator Name", value="sre_oncall")
            reason_text = st.text_input("Rollback Reason", value="Observed accuracy degradation on canary candidate")

            if st.button("🚨 Execute Authenticated Rollback", type="primary"):
                try:
                    resp = requests.post(
                        f"{API_BASE_URL}/rollback",
                        json={
                            "target_version": target_v,
                            "actor": operator_name,
                            "reason": reason_text,
                        },
                        timeout=5.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success(f"Rollback successful! Restored active model to: **{data['active_version']}**")
                        st.info("Smoke test passed: Verified inference output before promoting.")
                        st.rerun()
                    else:
                        st.error(f"Rollback failed: {resp.text}")
                except Exception as e:
                    st.error(f"Failed to communicate with API: {e}")

        with col_rb2:
            st.markdown("#### Deployment Audit History")
            events = db.query(DeploymentEvent).order_by(DeploymentEvent.timestamp.desc()).limit(15).all()
            if not events:
                st.info("No deployment events logged yet.")
            else:
                event_rows = [
                    {
                        "Timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        "Activated Version": e.activated_version,
                        "Previous Version": e.previous_version,
                        "Actor": e.actor,
                        "Reason": e.reason,
                        "Smoke Check Passed": "✅" if e.smoke_test_passed else "❌",
                    }
                    for e in events
                ]
                st.dataframe(pd.DataFrame(event_rows), use_container_width=True)

    # -------------------------------------------------------------------------
    # TAB 5: Live Request Logs
    # -------------------------------------------------------------------------
    with tab_logs:
        st.subheader("📋 Live Structured Prediction Request Logs")
        reqs = db.query(PredictionRecord).order_by(PredictionRecord.timestamp.desc()).limit(50).all()
        if not reqs:
            st.info("No prediction requests in database.")
        else:
            log_data = [
                {
                    "Request ID": r.request_id,
                    "Timestamp": r.timestamp.strftime("%H:%M:%S"),
                    "Model": r.model_version,
                    "Dist (km)": r.distance_km,
                    "Traffic": r.traffic_level,
                    "Weather": r.weather,
                    "Pred (min)": r.predicted_minutes,
                    "Latency (ms)": r.latency_ms,
                    "Status": r.status_code,
                }
                for r in reqs
            ]
            st.dataframe(pd.DataFrame(log_data), use_container_width=True)

    db.close()


if __name__ == "__main__":
    main()
