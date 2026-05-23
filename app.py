import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
FIGURE_DIR = APP_DIR / "figures"

RISK_ORDER = ["Critical", "High", "Watch", "Healthy"]
RISK_COLORS = {
    "Critical": "#d64545",
    "High": "#f28e2b",
    "Watch": "#f2c94c",
    "Healthy": "#2e8b57",
    "Anomaly": "#d64545",
    "Normal": "#2e8b57",
}

SCENARIO_RMSE = {
    "FD001": 18.4,
    "FD002": 21.7,
    "FD003": 19.6,
    "FD004": 22.1,
}

MODEL_BY_SCENARIO = {
    "FD001": {
        "en": "1D-CNN with SHAP review",
        "de": "1D-CNN mit SHAP-Review",
        "tr": "SHAP incelemeli 1D-CNN",
    },
    "FD002": {
        "en": "1D-CNN with multi-condition evaluation",
        "de": "1D-CNN mit Mehrbedingungen-Auswertung",
        "tr": "Çoklu çalışma koşulu değerlendirmeli 1D-CNN",
    },
    "FD003": {
        "en": "1D-CNN with two-fault-mode evaluation",
        "de": "1D-CNN mit Zwei-Fehlermodus-Auswertung",
        "tr": "İki arıza modu değerlendirmeli 1D-CNN",
    },
    "FD004": {
        "en": "1D-CNN with multi-condition/two-fault-mode evaluation",
        "de": "1D-CNN mit Mehrbedingungen- und Zwei-Fehlermodus-Auswertung",
        "tr": "Çoklu koşul / iki arıza modu değerlendirmeli 1D-CNN",
    },
}

current_lang = "en"

DISPLAY_LOCAL = {
    "en": {},
    "de": {
        "Critical": "Kritisch",
        "High": "Hoch",
        "High Risk": "Hohes Risiko",
        "Watch": "Beobachten",
        "Healthy": "Gesund",
        "Anomaly": "Anomalie",
        "Normal": "Normal",
        "Advanced degradation": "Fortgeschrittene Degradation",
        "Early degradation": "Frühe Degradation",
        "Stable validation signal": "Stabiles Validierungssignal",
        "Research-grade estimate": "Forschungsnahe Schätzung",
        "Experimental estimate": "Experimentelle Schätzung",
        "Elevated": "Erhöht",
        "Increasing": "Steigend",
        "Decreasing": "Fallend",
        "Stable": "Stabil",
        "Strong": "Stark",
        "Moderate": "Mittel",
        "Weak": "Schwach",
        "Escalating": "Eskalierend",
        "Severe escalation": "Starke Eskalation",
        "Medium": "Mittel",
        "Low": "Niedrig",
        "Urgent inspection": "Dringende Prüfung",
        "Schedule inspection review": "Prüfung einplanen",
        "Continue monitoring": "Weiter überwachen",
        "No immediate action": "Keine Sofortmaßnahme",
    },
    "tr": {
        "Critical": "Kritik",
        "High": "Yüksek",
        "High Risk": "Yüksek Risk",
        "Watch": "İzlemede",
        "Healthy": "Sağlıklı",
        "Anomaly": "Anomali",
        "Normal": "Normal",
        "Advanced degradation": "İleri bozulma",
        "Early degradation": "Erken bozulma",
        "Stable validation signal": "Stabil doğrulama sinyali",
        "Research-grade estimate": "Araştırma düzeyi tahmin",
        "Experimental estimate": "Deneysel tahmin",
        "Elevated": "Yükselmiş",
        "Increasing": "Artıyor",
        "Decreasing": "Azalıyor",
        "Stable": "Stabil",
        "Strong": "Güçlü",
        "Moderate": "Orta",
        "Weak": "Zayıf",
        "Escalating": "Tırmanıyor",
        "Severe escalation": "Şiddetli tırmanış",
        "Medium": "Orta",
        "Low": "Düşük",
        "Urgent inspection": "Acil inceleme",
        "Schedule inspection review": "İnceleme planla",
        "Continue monitoring": "İzlemeye devam et",
        "No immediate action": "Acil aksiyon yok",
    },
}


st.set_page_config(
    page_title="Predictive Maintenance Command Center",
    page_icon="🛠️",
    layout="wide",
)


def txt(en: str, de: str, tr_text: str) -> str:
    return {"en": en, "de": de, "tr": tr_text}.get(current_lang, en)


def tr(value: object) -> str:
    text = str(value)
    return DISPLAY_LOCAL.get(current_lang, {}).get(text, text)


def set_language(language: str) -> None:
    st.session_state["lang"] = language
    try:
        st.query_params["lang"] = language
    except Exception:
        pass


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cycle = pd.read_csv(DATA_DIR / "cmapss_engine_cycle_tableau.csv")
    fleet = pd.read_csv(DATA_DIR / "cmapss_fleet_summary_tableau.csv")
    scenario = pd.read_csv(DATA_DIR / "cmapss_scenario_summary_tableau.csv")
    risk = pd.read_csv(DATA_DIR / "cmapss_risk_band_counts_tableau.csv")
    bearing = pd.read_csv(DATA_DIR / "ims_bearing_rms_anomaly_tableau.csv", parse_dates=["timestamp"])
    return cycle, fleet, scenario, risk, bearing


def risk_from_rul(rul: float) -> str:
    if rul <= 30:
        return "Critical"
    if rul <= 60:
        return "High"
    if rul <= 125:
        return "Watch"
    return "Healthy"


def action_from_risk(risk: str) -> str:
    return {
        "Critical": txt(
            "Immediate review required. Prepare maintenance ticket and inspect asset.",
            "Sofortige Prüfung erforderlich. Wartungsticket vorbereiten und Asset prüfen.",
            "Acil inceleme gerekli. Bakım kaydı hazırlayın ve varlığı kontrol edin.",
        ),
        "High": txt(
            "Schedule inspection and monitor degradation trend closely.",
            "Prüfung einplanen und Degradationstrend eng überwachen.",
            "İnceleme planlayın ve bozulma trendini yakından izleyin.",
        ),
        "Watch": txt(
            "Continue monitoring and review again in the next planning cycle.",
            "Weiter überwachen und im nächsten Planungszyklus erneut prüfen.",
            "İzlemeye devam edin ve bir sonraki planlama döngüsünde tekrar değerlendirin.",
        ),
        "Healthy": txt(
            "No immediate action. Keep asset in routine monitoring.",
            "Keine Sofortmaßnahme. Asset weiter routinemäßig überwachen.",
            "Acil aksiyon yok. Varlığı rutin izleme altında tutun.",
        ),
    }[risk]


def queue_action(risk: str) -> str:
    return {
        "Critical": "Urgent inspection",
        "High": "Schedule inspection review",
        "Watch": "Continue monitoring",
        "Healthy": "No immediate action",
    }[risk]


def demo_predicted_rul(row: pd.Series) -> int:
    """Deterministic demo estimate, not live model inference."""
    phase = float(row["life_pct"])
    sensor_signal = (
        float(row.get("sensor_2", 0)) * 0.03
        + float(row.get("sensor_4", 0)) * 0.01
        - float(row.get("sensor_7", 0)) * 0.02
        + float(row.get("sensor_11", 0)) * 0.4
    )
    scenario_bias = {"FD001": -3, "FD002": 6, "FD003": -1, "FD004": 8}.get(row["scenario"], 0)
    phase_error = (phase - 0.55) * 18
    deterministic_error = scenario_bias + phase_error + (sensor_signal % 9) - 4
    return int(max(0, min(125, round(float(row["rul_capped_125"]) + deterministic_error))))


def prediction_reliability(error: float, scenario_name: str) -> str:
    rmse = SCENARIO_RMSE.get(scenario_name, 22)
    ratio = abs(error) / rmse
    if ratio <= 0.6:
        return "Stable validation signal"
    if ratio <= 1.1:
        return "Research-grade estimate"
    return "Experimental estimate"


def degradation_phase(life_pct: float, risk: str) -> str:
    if risk == "Critical" or life_pct >= 0.9:
        return "Critical"
    if risk == "High" or life_pct >= 0.7:
        return "Advanced degradation"
    if risk == "Watch" or life_pct >= 0.45:
        return "Early degradation"
    return "Healthy"


def fleet_at_inspection(df: pd.DataFrame, inspection_value: str) -> pd.DataFrame:
    if inspection_value == "Latest":
        snapshot = df.sort_values(["engine_id", "cycle"]).groupby("engine_id", as_index=False).tail(1)
    else:
        target_pct = int(inspection_value.replace("%", "")) / 100
        work = df.copy()
        work["inspection_distance"] = (work["life_pct"] - target_pct).abs()
        snapshot = (
            work.sort_values(["engine_id", "inspection_distance", "cycle"])
            .groupby("engine_id", as_index=False)
            .head(1)
            .drop(columns=["inspection_distance"])
        )

    snapshot = snapshot.copy()
    snapshot["current_risk"] = snapshot["rul"].apply(risk_from_rul)
    snapshot["predicted_rul"] = snapshot.apply(demo_predicted_rul, axis=1)
    snapshot["prediction_error"] = snapshot["predicted_rul"] - snapshot["rul"]
    snapshot["prediction_reliability"] = snapshot.apply(
        lambda row: prediction_reliability(row["prediction_error"], row["scenario"]),
        axis=1,
    )
    snapshot["degradation_phase"] = snapshot.apply(
        lambda row: degradation_phase(row["life_pct"], row["current_risk"]),
        axis=1,
    )
    return snapshot.sort_values("engine_id")


def fleet_health_timeline(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for point in ["30%", "50%", "70%", "90%", "Latest"]:
        snap = fleet_at_inspection(df, point)
        risk_counts = snap.groupby("current_risk").size().reset_index(name="engines")
        risk_counts["inspection_point"] = point
        frames.append(risk_counts)
    timeline = pd.concat(frames, ignore_index=True)
    timeline["current_risk"] = pd.Categorical(timeline["current_risk"], categories=RISK_ORDER, ordered=True)
    return timeline.sort_values(["inspection_point", "current_risk"])


def top_sensor_shift(engine_df: pd.DataFrame, selected_sensor: str) -> pd.DataFrame:
    numeric_sensors = [c for c in engine_df.columns if c.startswith("sensor_")]
    baseline = engine_df[numeric_sensors].head(min(20, len(engine_df))).mean()
    recent = engine_df[numeric_sensors].tail(min(20, len(engine_df))).mean()
    shift = (recent - baseline).abs().sort_values(ascending=False)
    explanation = pd.DataFrame(
        {
            "sensor": shift.head(6).index,
            "absolute_shift": shift.head(6).values.round(4),
            "dashboard_link": [
                txt("Selected sensor trend", "Trend des ausgewählten Sensors", "Seçili sensör trendi")
                if sensor == selected_sensor
                else txt("Supporting degradation signal", "Unterstützendes Degradationssignal", "Destekleyici bozulma sinyali")
                for sensor in shift.head(6).index
            ],
        }
    )
    return explanation


def max_sensor_shift(engine_df: pd.DataFrame) -> float:
    if len(engine_df) < 4:
        return 0.0
    numeric_sensors = [c for c in engine_df.columns if c.startswith("sensor_")]
    baseline = engine_df[numeric_sensors].head(min(20, len(engine_df))).mean()
    recent = engine_df[numeric_sensors].tail(min(20, len(engine_df))).mean()
    return float((recent - baseline).abs().max())


def alert_reason(row: pd.Series, max_shift: float) -> str:
    if row["rul"] <= 30:
        return txt("Low RUL and high life consumption", "Niedrige RUL und hoher Lebensdauerverbrauch", "Düşük RUL ve yüksek ömür tüketimi")
    if row["life_pct"] >= 0.75 and row["rul"] <= 60:
        return txt("High life consumption with limited RUL", "Hoher Lebensdauerverbrauch mit begrenzter RUL", "Yüksek ömür tüketimi ve sınırlı RUL")
    if max_shift > 5:
        return txt("Top sensor deviation detected", "Deutliche Sensorabweichung erkannt", "Belirgin sensör sapması tespit edildi")
    return txt("Risk band requires monitoring", "Risikoband erfordert Überwachung", "Risk bandı izleme gerektiriyor")


def engine_health_score(row: pd.Series, max_shift: float) -> int:
    rul_component = min(float(row["rul_capped_125"]) / 125, 1) * 70
    life_penalty = float(row["life_pct"]) * 15
    risk_penalty = {"Healthy": 0, "Watch": 8, "High": 18, "Critical": 32}[row["current_risk"]]
    sensor_penalty = min(max_shift / 20, 1) * 15
    score = rul_component + 30 - life_penalty - risk_penalty - sensor_penalty
    return int(max(0, min(100, round(score))))


def health_band(score: int, risk: str) -> str:
    if risk == "Critical":
        return "Critical"
    if risk == "High":
        return "High Risk"
    if risk == "Watch":
        return "Watch"
    if score < 75:
        return "Watch"
    return "Healthy"


def fleet_alerts(snapshot: pd.DataFrame, scenario_cycle: pd.DataFrame) -> pd.DataFrame:
    rows = []
    candidates = snapshot.sort_values(["predicted_rul", "rul"]).head(8)
    for _, row in candidates.iterrows():
        engine_history = scenario_cycle[
            (scenario_cycle["engine_id"] == row["engine_id"]) & (scenario_cycle["cycle"] <= row["cycle"])
        ]
        shift = max_sensor_shift(engine_history)
        if row["current_risk"] in ["Critical", "High"] or row["predicted_rul"] <= 70 or shift > 5:
            rows.append(
                {
                    "engine_id": row["engine_id"],
                    "predicted_rul": int(row["predicted_rul"]),
                    "actual_rul": int(row["rul"]),
                    "risk_status": row["current_risk"],
                    "reason": alert_reason(row, shift),
                }
            )
    return pd.DataFrame(rows).head(5)


def maintenance_queue(snapshot: pd.DataFrame) -> pd.DataFrame:
    rank = {"Critical": 1, "High": 2, "Watch": 3, "Healthy": 4}
    queue = snapshot.copy()
    queue["priority_rank"] = queue["current_risk"].map(rank)
    queue["recommended_action"] = queue["current_risk"].apply(queue_action)
    queue = queue.sort_values(["priority_rank", "predicted_rul", "rul"]).head(5)
    return queue[
        ["engine_id", "current_risk", "predicted_rul", "rul", "life_pct", "recommended_action"]
    ].rename(
        columns={
            "current_risk": "risk_status",
            "rul": "actual_rul",
        }
    )


def engine_story(engine_id: str, row: pd.Series, score: int, band: str) -> str:
    return txt(
        (
            f"Engine {engine_id} is currently in {tr(row['degradation_phase']).lower()}. "
            f"At the selected inspection point, actual RUL is {int(row['rul'])} cycles and demo predicted RUL is {int(row['predicted_rul'])} cycles. "
            f"The engine is classified as {tr(row['current_risk'])}. "
            f"The rule-based health score is {score}/100 ({tr(band)}); recommended action: {tr(queue_action(row['current_risk'])).lower()}."
        ),
        (
            f"Motor {engine_id} befindet sich aktuell in der Phase {tr(row['degradation_phase']).lower()}. "
            f"Am gewählten Prüfpunkt beträgt die tatsächliche RUL {int(row['rul'])} Zyklen und die Demo-Prognose {int(row['predicted_rul'])} Zyklen. "
            f"Der Motor wird als {tr(row['current_risk'])} eingestuft. "
            f"Der regelbasierte Health Score liegt bei {score}/100 ({tr(band)}); empfohlene Aktion: {tr(queue_action(row['current_risk'])).lower()}."
        ),
        (
            f"{engine_id} motoru şu anda {tr(row['degradation_phase']).lower()} fazında. "
            f"Seçili kontrol noktasında gerçek RUL {int(row['rul'])} çevrim, demo tahmini RUL ise {int(row['predicted_rul'])} çevrim. "
            f"Motorun risk durumu {tr(row['current_risk'])} olarak sınıflandırıldı. "
            f"Kural tabanlı sağlık skoru {score}/100 ({tr(band)}); önerilen aksiyon: {tr(queue_action(row['current_risk'])).lower()}."
        ),
    )


def status_chip(label: str, value: str) -> None:
    color = {
        "Critical": "#d64545",
        "High": "#f28e2b",
        "High Risk": "#f28e2b",
        "Watch": "#f2c94c",
        "Healthy": "#2e8b57",
        "Advanced degradation": "#f28e2b",
        "Early degradation": "#f2c94c",
        "Medium": "#f2c94c",
        "Low": "#d64545",
        "Stable validation signal": "#2e8b57",
        "Research-grade estimate": "#f2c94c",
        "Experimental estimate": "#f28e2b",
        "Elevated": "#f28e2b",
        "Increasing": "#f28e2b",
        "Decreasing": "#246bfe",
        "Stable": "#2e8b57",
        "Strong": "#d64545",
        "Moderate": "#f28e2b",
        "Weak": "#2e8b57",
        "Escalating": "#f28e2b",
        "Severe escalation": "#d64545",
    }.get(value, "#2e8b57")
    display_value = tr(value)
    st.markdown(
        f"""
        <div style="padding:0.72rem 0.85rem;border:1px solid rgba(127,127,127,.28);border-radius:10px;background:rgba(127,127,127,.06);">
          <div style="font-size:0.78rem;color:rgba(127,127,127,.95);margin-bottom:0.35rem;">{label}</div>
          <span style="display:inline-block;padding:0.22rem 0.58rem;border-radius:999px;background:{color}22;color:{color};font-weight:700;">
            {display_value}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def value_card(label: str, value: str, accent: str = "#246bfe") -> None:
    display_value = tr(value)
    st.markdown(
        f"""
        <div style="min-height:104px;padding:0.82rem 0.9rem;border:1px solid rgba(127,127,127,.24);border-radius:10px;background:rgba(127,127,127,.055);">
          <div style="font-size:0.76rem;color:rgba(127,127,127,.95);margin-bottom:0.45rem;">{label}</div>
          <div style="font-size:1.02rem;line-height:1.35;font-weight:760;color:{accent};white-space:normal;overflow-wrap:anywhere;">
            {display_value}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def compact_text_card(label: str, value: str, accent: str = "#2e8b57") -> None:
    display_value = tr(value)
    st.markdown(
        f"""
        <div title="{display_value}" style="padding:0.72rem 0.82rem;border:1px solid rgba(127,127,127,.28);border-radius:10px;background:rgba(127,127,127,.06);min-height:82px;">
          <div style="font-size:0.74rem;color:rgba(127,127,127,.95);margin-bottom:0.35rem;">{label}</div>
          <div style="font-size:0.92rem;line-height:1.25;font-weight:760;color:{accent};white-space:normal;overflow-wrap:break-word;">
            {display_value}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_kpi_strip(snapshot: pd.DataFrame) -> None:
    counts = snapshot["current_risk"].value_counts().to_dict()
    cols = st.columns(4)
    for col, risk_name in zip(cols, ["Healthy", "Watch", "High", "Critical"]):
        with col:
            value_card(
                txt(f"{tr(risk_name)} engines", f"{tr(risk_name)} Motoren", f"{tr(risk_name)} motor"),
                str(int(counts.get(risk_name, 0))),
                RISK_COLORS[risk_name],
            )


def executive_summary(snapshot: pd.DataFrame, inspection_point: str, impact: dict[str, str]) -> None:
    counts = snapshot["current_risk"].value_counts().to_dict()
    dominant_risk = snapshot["current_risk"].mode().iloc[0]
    critical_count = int(counts.get("Critical", 0))
    high_count = int(counts.get("High", 0))

    if critical_count:
        recommendation = txt(
            "Immediate inspection queue review recommended.",
            "Sofortige Prüfung der Inspektions-Warteschlange empfohlen.",
            "Acil inceleme kuyruğu gözden geçirilmeli.",
        )
        accent = RISK_COLORS["Critical"]
    elif high_count:
        recommendation = txt(
            "Inspection planning recommended for high-risk engines.",
            "Inspektionsplanung für Motoren mit hohem Risiko empfohlen.",
            "Yüksek riskli motorlar için inceleme planı önerilir.",
        )
        accent = RISK_COLORS["High"]
    else:
        recommendation = txt(
            "Monitoring recommended; no critical engines detected.",
            "Überwachung empfohlen; keine kritischen Motoren erkannt.",
            "İzleme önerilir; kritik motor tespit edilmedi.",
        )
        accent = RISK_COLORS.get(dominant_risk, "#246bfe")

    summary_title = txt("Executive summary", "Management-Zusammenfassung", "Yönetici özeti")
    summary_body = txt(
        f"Fleet operating in <span style='color:{accent};'>{tr(dominant_risk)}</span> state at the {inspection_point} inspection horizon. {critical_count} critical engines and {high_count} high-risk engines detected. {recommendation}",
        f"Die Flotte arbeitet am Prüfhorizont {inspection_point} im Zustand <span style='color:{accent};'>{tr(dominant_risk)}</span>. {critical_count} kritische Motoren und {high_count} Hochrisiko-Motoren erkannt. {recommendation}",
        f"Filo {inspection_point} kontrol ufkunda <span style='color:{accent};'>{tr(dominant_risk)}</span> durumunda çalışıyor. {critical_count} kritik motor ve {high_count} yüksek riskli motor tespit edildi. {recommendation}",
    )
    decision_signal = txt(
        f"Selected-engine decision signal: {tr(impact['Estimated maintenance priority'])} priority · {tr(impact['Downtime risk level'])} downtime risk · {tr(impact['Suggested action'])}.",
        f"Entscheidungssignal des ausgewählten Motors: {tr(impact['Estimated maintenance priority'])} Priorität · {tr(impact['Downtime risk level'])} Stillstandsrisiko · {tr(impact['Suggested action'])}.",
        f"Seçili motor karar sinyali: {tr(impact['Estimated maintenance priority'])} öncelik · {tr(impact['Downtime risk level'])} duruş riski · {tr(impact['Suggested action'])}.",
    )

    st.markdown(
        f"""
        <div style="padding:1rem 1.05rem;border:1px solid {accent}55;border-left:5px solid {accent};border-radius:12px;background:linear-gradient(90deg,{accent}20,rgba(127,127,127,.045));margin:0.55rem 0 1.05rem;">
          <div style="font-size:0.75rem;text-transform:uppercase;letter-spacing:.04em;color:rgba(127,127,127,.95);font-weight:800;margin-bottom:.32rem;">{summary_title}</div>
          <div style="font-size:1.08rem;line-height:1.42;font-weight:760;">
            {summary_body}
          </div>
          <div style="font-size:0.82rem;color:rgba(127,127,127,.95);margin-top:.42rem;">
            {decision_signal}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_card(scenario_name: str) -> None:
    rmse = SCENARIO_RMSE.get(scenario_name)
    model_name = MODEL_BY_SCENARIO.get(scenario_name, {}).get(current_lang, "1D-CNN")
    card = pd.DataFrame(
        [
            {
                txt("Model", "Modell", "Model"): model_name,
                txt("Scenario", "Szenario", "Senaryo"): scenario_name,
                txt("Window size", "Fenstergröße", "Pencere boyutu"): "20",
                txt("RUL cap", "RUL-Kappung", "RUL sınırı"): "125",
                txt("Validation", "Validierung", "Doğrulama"): txt("Engine-wise GroupKFold", "Motorweiser GroupKFold", "Motor bazlı GroupKFold"),
                "RMSE": f"{rmse:.1f}" if rmse else "18-22 thesis range",
            }
        ]
    )
    st.dataframe(card, width="stretch", hide_index=True)
    st.caption(txt(
        "Predicted RUL is a demo estimate from saved experiment artifacts, not live model inference.",
        "Die vorhergesagte RUL ist eine Demo-Schätzung aus gespeicherten Experimentartefakten, keine Live-Modellinferenz.",
        "Tahmini RUL, kaydedilmiş deney çıktılarından türetilen demo tahminidir; canlı model çıkarımı değildir.",
    ))


def pipeline_flow() -> None:
    st.markdown(
        f"""
        <div style="display:flex;gap:8px;flex-wrap:wrap;margin:0.35rem 0 1rem;align-items:center;">
          <span style="padding:0.45rem 0.7rem;border-radius:999px;background:rgba(15,143,116,.14);border:1px solid rgba(15,143,116,.35);font-weight:700;">{txt("Dataset", "Datensatz", "Veri Seti")}</span>
          <span>→</span>
          <span style="padding:0.45rem 0.7rem;border-radius:999px;background:rgba(15,143,116,.14);border:1px solid rgba(15,143,116,.35);font-weight:700;">{txt("Preprocessing", "Vorverarbeitung", "Ön İşleme")}</span>
          <span>→</span>
          <span style="padding:0.45rem 0.7rem;border-radius:999px;background:rgba(15,143,116,.14);border:1px solid rgba(15,143,116,.35);font-weight:700;">{txt("Sliding Window (20)", "Gleitendes Fenster (20)", "Kayan Pencere (20)")}</span>
          <span>→</span>
          <span style="padding:0.45rem 0.7rem;border-radius:999px;background:rgba(36,107,254,.14);border:1px solid rgba(36,107,254,.35);font-weight:700;">1D-CNN / XGBoost</span>
          <span>→</span>
          <span style="padding:0.45rem 0.7rem;border-radius:999px;background:rgba(36,107,254,.14);border:1px solid rgba(36,107,254,.35);font-weight:700;">{txt("Prediction", "Prognose", "Tahmin")}</span>
          <span>→</span>
          <span style="padding:0.45rem 0.7rem;border-radius:999px;background:rgba(242,142,43,.14);border:1px solid rgba(242,142,43,.35);font-weight:700;">SHAP</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sensor_comment(engine_df: pd.DataFrame, selected_sensor: str) -> str:
    if len(engine_df) < 4:
        return txt(
            "Not enough inspection history is available to comment on the selected sensor.",
            "Nicht genügend Prüfhistorie für eine belastbare Sensorkommentierung verfügbar.",
            "Seçili sensör için yorum üretmeye yetecek kontrol geçmişi yok.",
        )
    early = float(engine_df[selected_sensor].head(min(20, len(engine_df))).mean())
    recent = float(engine_df[selected_sensor].tail(min(20, len(engine_df))).mean())
    change = recent - early
    relative = abs(change) / max(abs(early), 1)
    direction = txt("increased", "gestiegen", "arttı") if change > 0 else txt("decreased", "gefallen", "azaldı")
    if relative < 0.002:
        return txt(
            f"{selected_sensor} is relatively stable up to this inspection point; it does not show a strong shift in this dashboard view.",
            f"{selected_sensor} ist bis zu diesem Prüfpunkt relativ stabil und zeigt in dieser Dashboard-Sicht keine starke Abweichung.",
            f"{selected_sensor} bu kontrol noktasına kadar görece stabil; bu dashboard görünümünde güçlü bir sapma göstermiyor.",
        )
    if relative < 0.01:
        return txt(
            f"{selected_sensor} has {direction} slightly up to this inspection point; treat it as a weak supporting signal rather than a standalone warning.",
            f"{selected_sensor} ist bis zu diesem Prüfpunkt leicht {direction}; als schwaches unterstützendes Signal interpretieren, nicht als alleinige Warnung.",
            f"{selected_sensor} bu kontrol noktasına kadar hafifçe {direction}; bunu tek başına uyarı değil, zayıf destekleyici sinyal olarak okuyun.",
        )
    return txt(
        f"{selected_sensor} has {direction} noticeably up to this inspection point; review it alongside RUL risk and SHAP context.",
        f"{selected_sensor} ist bis zu diesem Prüfpunkt deutlich {direction}; gemeinsam mit RUL-Risiko und SHAP-Kontext prüfen.",
        f"{selected_sensor} bu kontrol noktasına kadar belirgin şekilde {direction}; RUL riski ve SHAP bağlamıyla birlikte incelenmeli.",
    )


def sensor_signal_summary(engine_df: pd.DataFrame, selected_sensor: str) -> dict[str, str]:
    if len(engine_df) < 4:
        return {
            "Selected sensor": selected_sensor,
            "Trend type": "Insufficient history",
            "Deviation score": "0.00",
            "Deviation severity": "Weak",
            "Signal strength": "Weak",
            "Dashboard interpretation": txt(
                "Not enough inspection history is available for a reliable trend summary.",
                "Für eine zuverlässige Trendzusammenfassung ist nicht genügend Prüfhistorie vorhanden.",
                "Güvenilir trend özeti için yeterli kontrol geçmişi yok.",
            ),
        }

    early = float(engine_df[selected_sensor].head(min(20, len(engine_df))).mean())
    recent = float(engine_df[selected_sensor].tail(min(20, len(engine_df))).mean())
    change = recent - early
    relative = abs(change) / max(abs(early), 1)
    deviation_score = relative * 100

    if relative < 0.002:
        trend = "Stable"
        strength = "Weak"
        severity = "Weak"
    elif change > 0:
        trend = "Increasing"
        strength = "Moderate" if relative < 0.01 else "Strong"
        severity = strength
    else:
        trend = "Decreasing"
        strength = "Moderate" if relative < 0.01 else "Strong"
        severity = strength

    return {
        "Selected sensor": selected_sensor,
        "Trend type": trend,
        "Deviation score": f"{deviation_score:.2f}",
        "Deviation severity": severity,
        "Signal strength": strength,
        "Dashboard interpretation": sensor_comment(engine_df, selected_sensor),
    }


def business_impact(row: pd.Series, max_shift: float) -> dict[str, str]:
    risk = row["current_risk"]
    life_pct = float(row["life_pct"])
    predicted_rul = float(row["predicted_rul"])

    if risk == "Critical" or predicted_rul <= 30:
        priority = "Critical"
        downtime_risk = "Critical"
    elif risk == "High" or predicted_rul <= 60:
        priority = "High"
        downtime_risk = "High"
    elif risk == "Watch" or life_pct >= 0.55 or max_shift > 5:
        priority = "Medium"
        downtime_risk = "Elevated"
    else:
        priority = "Low"
        downtime_risk = "Low"

    if max_shift > 5:
        sensor_basis = txt("strong sensor deviation", "starke Sensorabweichung", "güçlü sensör sapması")
    elif max_shift > 1:
        sensor_basis = txt("moderate sensor deviation", "mittlere Sensorabweichung", "orta düzey sensör sapması")
    else:
        sensor_basis = txt("limited sensor deviation", "begrenzte Sensorabweichung", "sınırlı sensör sapması")

    return {
        "Estimated maintenance priority": priority,
        "Downtime risk level": downtime_risk,
        "Suggested action": queue_action(risk),
        "Basis": txt(
            f"{tr(risk)} risk band + {life_pct * 100:.1f}% life consumed + {sensor_basis} + predicted RUL {int(predicted_rul)} cycles",
            f"{tr(risk)} Risikoband + {life_pct * 100:.1f}% Lebensdauer verbraucht + {sensor_basis} + prognostizierte RUL {int(predicted_rul)} Zyklen",
            f"{tr(risk)} risk bandı + %{life_pct * 100:.1f} ömür tüketimi + {sensor_basis} + tahmini RUL {int(predicted_rul)} çevrim",
        ),
    }


def kpi(label: str, value: str, help_text: str | None = None) -> None:
    st.metric(label, value, help=help_text)


def risk_badge(label: str) -> str:
    return f":{ {'Critical':'red','High':'orange','Watch':'orange','Healthy':'green'}.get(label, 'blue') }[{label}]"


def line_rul(df: pd.DataFrame, title: str) -> go.Figure:
    fig = px.line(
        df,
        x="cycle",
        y="rul",
        color="risk_band",
        color_discrete_map=RISK_COLORS,
        category_orders={"risk_band": RISK_ORDER},
        title=title,
    )
    for trace in fig.data:
        trace.name = tr(trace.name)
    fig.update_layout(legend_title_text=txt("Risk band", "Risikoband", "Risk bandı"), height=390, margin=dict(l=20, r=20, t=55, b=20))
    fig.update_yaxes(title=txt("Remaining Useful Life", "Restnutzungsdauer", "Kalan Faydalı Ömür"))
    fig.update_xaxes(title=txt("Cycle", "Zyklus", "Çevrim"))
    return fig


def sensor_chart(df: pd.DataFrame, sensor: str, engine_id: str) -> go.Figure:
    fig = px.line(df, x="cycle", y=sensor, title=txt(
        f"{engine_id} · {sensor.replace('_', ' ').title()} Trend",
        f"{engine_id} · {sensor.replace('_', ' ').title()} Trend",
        f"{engine_id} · {sensor.replace('_', ' ').title()} Trendi",
    ))
    fig.update_traces(line=dict(color="#246bfe", width=2))
    fig.update_layout(height=360, margin=dict(l=20, r=20, t=55, b=20))
    return fig


def top_sensor_bar(explanation: pd.DataFrame, engine_id: str) -> go.Figure:
    chart = explanation.sort_values("absolute_shift", ascending=True)
    fig = px.bar(
        chart,
        x="absolute_shift",
        y="sensor",
        orientation="h",
        color="absolute_shift",
        color_continuous_scale=["#1f8f74", "#f2c94c", "#f28e2b"],
        title=txt(
            f"{engine_id} · Top Sensor Shift",
            f"{engine_id} · Wichtigste Sensorverschiebungen",
            f"{engine_id} · En Belirgin Sensör Kaymaları",
        ),
    )
    fig.update_layout(
        height=285,
        margin=dict(l=20, r=20, t=55, b=20),
        coloraxis_showscale=False,
        yaxis_title=None,
        xaxis_title=txt("Absolute shift", "Absolute Verschiebung", "Mutlak kayma"),
    )
    return fig


def sensor_badge_row(sensor_signal: dict[str, str]) -> None:
    row1 = st.columns(2)
    with row1[0]:
        status_chip(txt("Selected sensor", "Ausgewählter Sensor", "Seçili sensör"), sensor_signal["Selected sensor"])
    with row1[1]:
        status_chip(txt("Trend type", "Trendtyp", "Trend tipi"), sensor_signal["Trend type"])

    row2 = st.columns(2)
    with row2[0]:
        status_chip(txt("Signal strength", "Signalstärke", "Sinyal gücü"), sensor_signal["Signal strength"])
    with row2[1]:
        status_chip(txt("Deviation severity", "Abweichungsstärke", "Sapma şiddeti"), sensor_signal["Deviation severity"])


def image_card(image_path: Path, caption: str) -> None:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <div style="padding:.72rem;border:1px solid rgba(127,127,127,.24);border-radius:12px;background:rgba(127,127,127,.06);box-shadow:0 10px 28px rgba(0,0,0,.10);">
          <div style="border-radius:9px;overflow:hidden;background:rgba(255,255,255,.06);">
            <img src="data:image/png;base64,{encoded}" style="display:block;width:100%;opacity:.92;filter:contrast(.96) brightness(.92);" />
          </div>
          <div style="font-size:.78rem;color:rgba(127,127,127,.95);margin-top:.55rem;">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def bearing_operational_context(
    bearing_df: pd.DataFrame,
    bearing_id: str,
    anomaly_count: int,
    threshold: float,
    max_rms: float,
) -> dict[str, object]:
    rms_col = f"{bearing_id}_rms"
    anomaly_rate = anomaly_count / max(len(bearing_df), 1)
    window = max(10, int(len(bearing_df) * 0.12))
    early_mean = float(bearing_df[rms_col].head(window).mean())
    recent_mean = float(bearing_df[rms_col].tail(window).mean())
    trend_ratio = recent_mean / max(early_mean, 1e-6)
    threshold_ratio = max_rms / max(threshold, 1e-6)

    if anomaly_count == 0:
        severity = "Healthy"
        progression = "Stable"
        urgency = "Low"
        action = "No immediate action"
    elif anomaly_rate >= 0.18 or threshold_ratio >= 2.0 or trend_ratio >= 1.8:
        severity = "Critical"
        progression = "Severe escalation"
        urgency = "Critical"
        action = "Urgent inspection"
    elif anomaly_rate >= 0.05 or threshold_ratio >= 1.3 or trend_ratio >= 1.25:
        severity = "High"
        progression = "Escalating"
        urgency = "High"
        action = "Schedule inspection review"
    else:
        severity = "Watch"
        progression = "Stable"
        urgency = "Medium"
        action = "Continue monitoring"

    basis = txt(
        f"RMS trend + threshold exceedance + {anomaly_rate * 100:.1f}% anomaly rate",
        f"RMS-Trend + Schwellenüberschreitung + {anomaly_rate * 100:.1f}% Anomalierate",
        f"RMS trendi + eşik aşımı + %{anomaly_rate * 100:.1f} anomali oranı",
    )
    return {
        "anomaly_rate": anomaly_rate,
        "trend_ratio": trend_ratio,
        "threshold_ratio": threshold_ratio,
        "severity": severity,
        "progression": progression,
        "maintenance_urgency": urgency,
        "recommended_action": action,
        "basis": basis,
    }


def bearing_summary_text(bearing_id: str, context: dict[str, object], anomaly_count: int) -> str:
    anomaly_rate = float(context["anomaly_rate"]) * 100
    return txt(
        (
            f"{bearing_id.replace('_', ' ').title()} shows {tr(context['progression']).lower()} behavior. "
            f"Anomalies account for {anomaly_rate:.1f}% of samples, with severity classified as {tr(context['severity'])}. "
            f"Recommended action: {tr(context['recommended_action']).lower()}."
        ),
        (
            f"{bearing_id.replace('_', ' ').title()} zeigt {tr(context['progression']).lower()} Verhalten. "
            f"Anomalien machen {anomaly_rate:.1f}% der Samples aus, die Schwere wird als {tr(context['severity'])} eingestuft. "
            f"Empfohlene Aktion: {tr(context['recommended_action']).lower()}."
        ),
        (
            f"{bearing_id.replace('_', ' ').title()} {tr(context['progression']).lower()} davranış gösteriyor. "
            f"Anomaliler örneklerin %{anomaly_rate:.1f} kısmını oluşturuyor; şiddet {tr(context['severity'])} olarak sınıflandırıldı. "
            f"Önerilen aksiyon: {tr(context['recommended_action']).lower()}."
        ),
    )


def bearing_chart(bearing: pd.DataFrame, bearing_id: str) -> go.Figure:
    rms_col = f"{bearing_id}_rms"
    threshold_col = f"{bearing_id}_threshold"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=bearing["timestamp"],
            y=bearing[rms_col],
            mode="lines",
            name="RMS",
            line=dict(color="#246bfe", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bearing["timestamp"],
            y=bearing[threshold_col],
            mode="lines",
            name=txt("Threshold", "Schwelle", "Eşik"),
            line=dict(color="#d64545", width=2, dash="dash"),
        )
    )
    anomaly = bearing[bearing[f"{bearing_id}_anomaly_flag"]]
    fig.add_trace(
        go.Scatter(
            x=anomaly["timestamp"],
            y=anomaly[rms_col],
            mode="markers",
            name=txt("Anomaly", "Anomalie", "Anomali"),
            marker=dict(color="#d64545", size=7),
        )
    )
    fig.update_layout(
        title=txt(
            f"{bearing_id.replace('_', ' ').title()} RMS Monitoring",
            f"{bearing_id.replace('_', ' ').title()} RMS-Überwachung",
            f"{bearing_id.replace('_', ' ').title()} RMS İzleme",
        ),
        height=390,
        margin=dict(l=20, r=20, t=55, b=20),
    )
    fig.update_yaxes(title="RMS")
    return fig


def build_alerts(cycle: pd.DataFrame, bearing: pd.DataFrame) -> pd.DataFrame:
    latest = cycle.sort_values(["engine_id", "cycle"]).groupby("engine_id", as_index=False).tail(1)
    engine_alerts = latest[latest["rul"] <= 60].copy()
    engine_alerts["asset_id"] = engine_alerts["engine_id"]
    engine_alerts["alert_type"] = "Düşük RUL"
    engine_alerts["severity"] = engine_alerts["rul"].apply(risk_from_rul)
    engine_alerts["trigger"] = "RUL = " + engine_alerts["rul"].round(0).astype(int).astype(str) + " çevrim"
    engine_alerts["recommended_action"] = engine_alerts["severity"].apply(action_from_risk)
    engine_alerts = engine_alerts[["asset_id", "scenario", "alert_type", "severity", "trigger", "recommended_action"]]

    bearing_rows = []
    for bearing_id in ["bearing_1", "bearing_2", "bearing_3", "bearing_4"]:
        flag = f"{bearing_id}_anomaly_flag"
        flagged = bearing[bearing[flag]].tail(1)
        if not flagged.empty:
            last = flagged.iloc[0]
            bearing_rows.append(
                {
                    "asset_id": bearing_id.replace("_", " ").title(),
                    "scenario": "IMS Bearing",
                    "alert_type": "Titreşim anomalisi",
                    "severity": "Critical",
                    "trigger": f"RMS {last['timestamp']} zamanında eşiği geçti",
                    "recommended_action": "Titreşim trendini inceleyin ve bakım değerlendirmesi hazırlayın.",
                }
            )

    bearing_alerts = pd.DataFrame(bearing_rows)
    alerts = pd.concat([engine_alerts, bearing_alerts], ignore_index=True)
    severity_rank = {"Critical": 1, "High": 2, "Watch": 3, "Healthy": 4}
    alerts["severity_rank"] = alerts["severity"].map(severity_rank).fillna(9)
    return alerts.sort_values(["severity_rank", "asset_id"]).drop(columns=["severity_rank"])


cycle, fleet, scenario, risk, bearing = load_data()

query_lang = st.query_params.get("lang", "en")
if isinstance(query_lang, list):
    query_lang = query_lang[0]
if "lang" not in st.session_state:
    st.session_state["lang"] = query_lang if query_lang in ["en", "de", "tr"] else "en"
current_lang = st.session_state["lang"]

with st.sidebar:
    selected_language = st.segmented_control(
        txt("Language", "Sprache", "Dil"),
        options=["en", "de", "tr"],
        format_func=lambda value: {"en": "🇬🇧 ENG", "de": "🇩🇪 DE", "tr": "🇹🇷 TR"}[value],
        default=current_lang,
        key="language_selector",
    )
    if selected_language and selected_language != current_lang:
        set_language(selected_language)
        st.rerun()
    st.divider()
    st.header(txt("Analysis Mode", "Analysemodus", "Analiz Modu"))
    analysis_mode = st.radio(
        txt("Choose analysis type", "Analysetyp auswählen", "Analiz tipini seç"),
        ["turbofan", "bearing"],
        format_func=lambda value: txt(
            "Turbofan RUL Prediction (NASA C-MAPSS)" if value == "turbofan" else "Bearing Anomaly Detection (IMS Bearing)",
            "Turbofan-RUL-Prognose (NASA C-MAPSS)" if value == "turbofan" else "Wälzlager-Anomalieerkennung (IMS Bearing)",
            "Turbofan RUL Tahmini (NASA C-MAPSS)" if value == "turbofan" else "Rulman Anomali Tespiti (IMS Bearing)",
        ),
    )

st.title(txt("Predictive Maintenance Command Center", "Predictive Maintenance Command Center", "Kestirimci Bakım Komuta Merkezi"))
st.caption(txt(
    "MVP demo with two focused analysis modes: turbofan RUL prediction and bearing anomaly detection.",
    "MVP-Demo mit zwei fokussierten Analysemodi: Turbofan-RUL-Prognose und Wälzlager-Anomalieerkennung.",
    "Turbofan RUL tahmini ve rulman anomali tespiti için iki odaklı MVP demosu.",
))

if analysis_mode == "turbofan":
    with st.sidebar:
        st.divider()
        st.header(txt("NASA Controls", "NASA-Steuerung", "NASA Kontrolleri"))
        selected_scenario = st.selectbox(txt("Scenario", "Szenario", "Senaryo"), sorted(cycle["scenario"].unique()))
        scenario_cycle = cycle[cycle["scenario"] == selected_scenario]
        inspection_point = st.selectbox(
            txt("Inspection point", "Prüfpunkt", "Kontrol noktası"),
            ["30%", "50%", "70%", "90%", "Latest"],
            index=2,
            format_func=lambda value: txt("Latest", "Aktuellster", "En güncel") if value == "Latest" else value,
        )
        scenario_snapshot = fleet_at_inspection(scenario_cycle, inspection_point)
        engine_options = sorted(scenario_snapshot["engine_id"].unique())
        selected_engine = st.selectbox(txt("Engine ID", "Motor-ID", "Motor ID"), engine_options)
        sensor_options = [c for c in cycle.columns if c.startswith("sensor_")]
        selected_sensor = st.selectbox(txt("Sensor", "Sensor", "Sensör"), sensor_options, index=sensor_options.index("sensor_7"))
        st.divider()
        with st.expander(txt("How to use", "So nutzen Sie es", "Nasıl kullanılır?"), expanded=False):
            st.caption(txt("1. Select scenario", "1. Szenario auswählen", "1. Senaryoyu seç"))
            st.caption(txt("2. Select inspection point", "2. Prüfpunkt auswählen", "2. Kontrol noktasını seç"))
            st.caption(txt("3. Select engine", "3. Motor auswählen", "3. Motoru seç"))
            st.caption(txt("4. Inspect sensor behavior", "4. Sensorverhalten prüfen", "4. Sensör davranışını incele"))
        st.caption(txt(
            "Tech Stack: Python · Streamlit · TensorFlow/Keras · Plotly · SHAP · GroupKFold",
            "Tech Stack: Python · Streamlit · TensorFlow/Keras · Plotly · SHAP · GroupKFold",
            "Teknoloji: Python · Streamlit · TensorFlow/Keras · Plotly · SHAP · GroupKFold",
        ))

    st.info(txt(
        "This mode estimates Remaining Useful Life (RUL) for turbofan engines and monitors fleet-level risk.",
        "Dieser Modus schätzt die Restnutzungsdauer (RUL) von Turbofan-Motoren und überwacht das Flottenrisiko.",
        "Bu mod turbofan motorları için kalan faydalı ömür (RUL) tahmini yapar ve filo seviyesinde riski izler.",
    ))

    scenario_meta = scenario[scenario["scenario"] == selected_scenario].iloc[0]
    engine_full_df = scenario_cycle[scenario_cycle["engine_id"] == selected_engine].copy().sort_values("cycle")
    current = scenario_snapshot[scenario_snapshot["engine_id"] == selected_engine].iloc[0]
    current_cycle = int(current["cycle"])
    engine_df = engine_full_df[engine_full_df["cycle"] <= current_cycle].copy()
    current_risk = risk_from_rul(float(current["rul"]))
    actual_rul = int(current["rul"])
    predicted_rul = int(current["predicted_rul"])
    prediction_error = int(current["prediction_error"])
    reliability = current["prediction_reliability"]
    phase = current["degradation_phase"]
    explanation = top_sensor_shift(engine_df, selected_sensor)
    selected_max_shift = max_sensor_shift(engine_df)
    selected_health_score = engine_health_score(current, selected_max_shift)
    selected_health_band = health_band(selected_health_score, current_risk)
    sensor_signal = sensor_signal_summary(engine_df, selected_sensor)
    impact = business_impact(current, selected_max_shift)

    executive_summary(scenario_snapshot, inspection_point, impact)

    st.subheader(txt("Overview", "Übersicht", "Genel Bakış"))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi(txt("Scenario", "Szenario", "Senaryo"), selected_scenario)
    with c2:
        kpi(txt("Inspection point", "Prüfpunkt", "Kontrol noktası"), txt("Latest", "Aktuellster", "En güncel") if inspection_point == "Latest" else inspection_point)
    with c3:
        kpi(txt("Engines evaluated", "Bewertete Motoren", "Değerlendirilen motor"), f"{scenario_snapshot['engine_id'].nunique():,}")
    with c4:
        kpi(txt("Avg RUL at inspection", "Ø RUL am Prüfpunkt", "Kontroldeki ort. RUL"), f"{scenario_snapshot['rul'].mean():.1f}")
    risk_kpi_strip(scenario_snapshot)

    st.subheader(txt("Active Alerts", "Aktive Warnungen", "Aktif Uyarılar"))
    st.caption(txt(
        "What needs attention right now? Alerts are rule-based and derived from selected inspection RUL, risk, life consumption, and sensor shift.",
        "Was braucht jetzt Aufmerksamkeit? Warnungen sind regelbasiert und leiten sich aus RUL, Risiko, Lebensdauerverbrauch und Sensorverschiebung ab.",
        "Şu anda ne dikkat istiyor? Uyarılar seçili kontrol RUL'u, risk, ömür tüketimi ve sensör kaymasına göre kural tabanlı üretilir.",
    ))
    alerts = fleet_alerts(scenario_snapshot, scenario_cycle)
    if alerts.empty:
        st.success(txt(
            "No active high-priority engine alerts at this inspection point.",
            "Keine aktiven hochpriorisierten Motorwarnungen an diesem Prüfpunkt.",
            "Bu kontrol noktasında aktif yüksek öncelikli motor uyarısı yok.",
        ))
    else:
        alerts = alerts.copy()
        alerts["risk_status"] = alerts["risk_status"].map(tr)
        st.dataframe(
            alerts.rename(
                columns={
                    "engine_id": txt("engine_id", "motor_id", "motor_id"),
                    "predicted_rul": txt("predicted_rul", "prognose_rul", "tahmini_rul"),
                    "actual_rul": txt("actual_rul", "tatsächliche_rul", "gerçek_rul"),
                    "risk_status": txt("risk_status", "risikostatus", "risk_durumu"),
                    "reason": txt("reason", "grund", "neden"),
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.subheader(txt("Model Card", "Modellkarte", "Model Kartı"))
    model_card(selected_scenario)

    st.subheader(txt("Pipeline", "Pipeline", "İş Akışı"))
    pipeline_flow()

    left, right = st.columns([1.1, 1])
    with left:
        risk_view = (
            scenario_snapshot.groupby("current_risk")
            .size()
            .reset_index(name="engines")
        )
        fig = px.bar(
            risk_view,
            x="current_risk",
            y="engines",
            color="current_risk",
            color_discrete_map=RISK_COLORS,
            category_orders={"current_risk": RISK_ORDER},
            title=txt(
                f"{selected_scenario} Risk Distribution at {inspection_point}",
                f"{selected_scenario} Risikoverteilung am Prüfpunkt {inspection_point}",
                f"{selected_scenario} · {inspection_point} Kontrolünde Risk Dağılımı",
            ),
        )
        fig.update_xaxes(title=txt("Risk status", "Risikostatus", "Risk durumu"), tickvals=RISK_ORDER, ticktext=[tr(item) for item in RISK_ORDER])
        fig.update_yaxes(title=txt("Engine count", "Motoranzahl", "Motor sayısı"))
        fig.update_layout(height=360, margin=dict(l=20, r=20, t=55, b=20), showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with right:
        priority = scenario_snapshot.sort_values("rul").head(10)[["engine_id", "cycle", "max_cycle", "rul", "life_pct", "current_risk"]]
        priority["life_pct"] = (priority["life_pct"] * 100).round(1).astype(str) + "%"
        priority["current_risk"] = priority["current_risk"].map(tr)
        st.markdown(txt(
            f"#### Highest Priority Engines · {selected_scenario} · {inspection_point}",
            f"#### Motoren mit höchster Priorität · {selected_scenario} · {inspection_point}",
            f"#### En Öncelikli Motorlar · {selected_scenario} · {inspection_point}",
        ))
        st.dataframe(
            priority.rename(
                columns={
                    "engine_id": txt("engine_id", "motor_id", "motor_id"),
                    "cycle": txt("cycle", "zyklus", "çevrim"),
                    "max_cycle": txt("max_cycle", "max_zyklus", "maks_çevrim"),
                    "rul": "rul",
                    "life_pct": txt("life_consumed", "lebensdauer_verbraucht", "ömür_tüketimi"),
                    "current_risk": txt("risk_status", "risikostatus", "risk_durumu"),
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.subheader(txt("Fleet Health Timeline", "Flottenzustand im Zeitverlauf", "Filo Sağlık Zaman Çizgisi"))
    timeline = fleet_health_timeline(scenario_cycle)
    fig = px.bar(
        timeline,
        x="inspection_point",
        y="engines",
        color="current_risk",
        color_discrete_map=RISK_COLORS,
        category_orders={"current_risk": RISK_ORDER, "inspection_point": ["30%", "50%", "70%", "90%", "Latest"]},
        title=txt(
            f"{selected_scenario} Fleet Degradation Progression",
            f"{selected_scenario} Flotten-Degradationsverlauf",
            f"{selected_scenario} Filo Bozulma İlerlemesi",
        ),
    )
    for trace in fig.data:
        trace.name = tr(trace.name)
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=55, b=20), legend_title_text=txt("Risk", "Risiko", "Risk"))
    fig.update_xaxes(title=txt("Inspection point", "Prüfpunkt", "Kontrol noktası"))
    fig.update_yaxes(title=txt("Engine count", "Motoranzahl", "Motor sayısı"))
    st.plotly_chart(fig, width="stretch")

    st.subheader(txt("Engine Detail", "Motordetail", "Motor Detayı"))
    st.caption(txt(
        f"Selected engine: {selected_engine} · Inspection point: {inspection_point}",
        f"Ausgewählter Motor: {selected_engine} · Prüfpunkt: {inspection_point}",
        f"Seçili motor: {selected_engine} · Kontrol noktası: {inspection_point}",
    ))
    score_col, metric_col = st.columns([0.9, 2.1])
    with score_col:
        st.metric(txt("Engine Health Score", "Motor Health Score", "Motor Sağlık Skoru"), f"{selected_health_score}/100")
        status_chip(txt("Health band", "Health-Band", "Sağlık bandı"), selected_health_band)
        st.caption(txt(
            "Rule-based dashboard score using RUL, risk band, life consumed, and sensor shift.",
            "Regelbasierter Dashboard-Score aus RUL, Risikoband, Lebensdauerverbrauch und Sensorverschiebung.",
            "RUL, risk bandı, ömür tüketimi ve sensör kaymasına dayalı kural tabanlı dashboard skoru.",
        ))
    with metric_col:
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            kpi(txt("Current cycle", "Aktueller Zyklus", "Mevcut çevrim"), f"{current_cycle:,}")
        with e2:
            kpi(txt("Actual RUL at inspection", "Tatsächliche RUL am Prüfpunkt", "Kontroldeki gerçek RUL"), txt(f"{actual_rul} cycles", f"{actual_rul} Zyklen", f"{actual_rul} çevrim"))
        with e3:
            kpi(txt("Predicted RUL", "Prognostizierte RUL", "Tahmini RUL"), txt(f"{predicted_rul} cycles", f"{predicted_rul} Zyklen", f"{predicted_rul} çevrim"))
        with e4:
            kpi(txt("Prediction error", "Prognosefehler", "Tahmin hatası"), txt(f"{prediction_error:+d} cycles", f"{prediction_error:+d} Zyklen", f"{prediction_error:+d} çevrim"))

        e5, e6, e7 = st.columns(3)
        with e5:
            compact_text_card(txt("Validation signal", "Validierungssignal", "Doğrulama sinyali"), reliability)
        with e6:
            status_chip(txt("Risk status", "Risikostatus", "Risk durumu"), current_risk)
        with e7:
            status_chip(txt("Degradation phase", "Degradationsphase", "Bozulma fazı"), phase)

    st.warning(txt("Maintenance recommendation: ", "Wartungsempfehlung: ", "Bakım önerisi: ") + action_from_risk(current_risk))
    st.caption(txt(
        "Predicted RUL is a demo estimate based on saved experiment artifacts; Actual RUL comes from the labeled C-MAPSS run-to-failure data.",
        "Die vorhergesagte RUL ist eine Demo-Schätzung aus gespeicherten Experimentartefakten; die tatsächliche RUL stammt aus den gelabelten C-MAPSS Run-to-Failure-Daten.",
        "Tahmini RUL, kaydedilmiş deney çıktılarından türetilen demo tahminidir; gerçek RUL etiketli C-MAPSS run-to-failure verisinden gelir.",
    ))
    st.caption(txt(
        "Validation signal is based on saved experiment performance and engine-wise validation artifacts; it is not calibrated production confidence.",
        "Das Validierungssignal basiert auf gespeicherter Experimentleistung und motorweisen Validierungsartefakten; es ist kein kalibriertes Produktions-Konfidenzmaß.",
        "Doğrulama sinyali, kaydedilmiş deney performansı ve motor bazlı doğrulama çıktıları üzerinden verilir; kalibre edilmiş üretim güven skoru değildir.",
    ))

    st.subheader(txt("Engine Story / Summary", "Motorstory / Zusammenfassung", "Motor Hikayesi / Özet"))
    st.write(engine_story(selected_engine, current, selected_health_score, selected_health_band))

    st.markdown(txt("#### Business Impact", "#### Geschäftsauswirkung", "#### İş Etkisi"))
    impact_cols = st.columns(3)
    with impact_cols[0]:
        value_card(txt("Maintenance priority", "Wartungspriorität", "Bakım önceliği"), impact["Estimated maintenance priority"], RISK_COLORS.get(impact["Estimated maintenance priority"], "#246bfe"))
    with impact_cols[1]:
        value_card(txt("Downtime risk level", "Stillstandsrisiko", "Duruş riski seviyesi"), impact["Downtime risk level"], RISK_COLORS.get(impact["Downtime risk level"], "#f28e2b"))
    with impact_cols[2]:
        value_card(txt("Suggested action", "Empfohlene Aktion", "Önerilen aksiyon"), impact["Suggested action"], "#246bfe")
    st.caption(txt(
        f"Basis: {impact['Basis']}. Rule-based decision support only; no downtime duration or cost estimate is inferred.",
        f"Basis: {impact['Basis']}. Nur regelbasierte Entscheidungsunterstützung; es wird keine Stillstands- oder Kostenschätzung abgeleitet.",
        f"Dayanak: {impact['Basis']}. Yalnızca kural tabanlı karar desteğidir; duruş süresi veya maliyet tahmini yapılmaz.",
    ))

    st.markdown(txt("#### Fleet Comparison", "#### Flottenvergleich", "#### Filo Karşılaştırması"))
    dominant_risk = scenario_snapshot["current_risk"].mode().iloc[0]
    comparison = pd.DataFrame(
        [
            {
                txt("metric", "metrik", "metrik"): "RUL",
                txt("selected_engine", "ausgewählter_motor", "seçili_motor"): txt(f"{actual_rul} cycles", f"{actual_rul} Zyklen", f"{actual_rul} çevrim"),
                txt("fleet_average", "flottendurchschnitt", "filo_ortalaması"): txt(f"{float(scenario_snapshot['rul'].mean()):.1f} cycles", f"{float(scenario_snapshot['rul'].mean()):.1f} Zyklen", f"{float(scenario_snapshot['rul'].mean()):.1f} çevrim"),
                txt("interpretation", "interpretation", "yorum"): txt("Lower is more urgent", "Niedriger ist dringender", "Daha düşük değer daha acil durumu gösterir"),
            },
            {
                txt("metric", "metrik", "metrik"): txt("Life consumed %", "Lebensdauer verbraucht %", "Tüketilen ömür %"),
                txt("selected_engine", "ausgewählter_motor", "seçili_motor"): f"{float(current['life_pct'] * 100):.1f}%",
                txt("fleet_average", "flottendurchschnitt", "filo_ortalaması"): f"{float(scenario_snapshot['life_pct'].mean() * 100):.1f}%",
                txt("interpretation", "interpretation", "yorum"): txt("Higher means later lifecycle", "Höher bedeutet später im Lebenszyklus", "Daha yüksek değer yaşam döngüsünde daha geç aşamayı gösterir"),
            },
            {
                txt("metric", "metrik", "metrik"): txt("Risk", "Risiko", "Risk"),
                txt("selected_engine", "ausgewählter_motor", "seçili_motor"): tr(current_risk),
                txt("fleet_average", "flottendurchschnitt", "filo_ortalaması"): txt(f"Dominant: {tr(dominant_risk)}", f"Dominant: {tr(dominant_risk)}", f"En yaygın: {tr(dominant_risk)}"),
                txt("interpretation", "interpretation", "yorum"): txt("Compares selected engine to fleet state", "Vergleicht den ausgewählten Motor mit dem Flottenzustand", "Seçili motoru filo geneliyle karşılaştırır"),
            },
        ]
    )
    st.dataframe(comparison, width="stretch", hide_index=True)

    st.subheader(txt("Maintenance Queue", "Wartungswarteschlange", "Bakım Kuyruğu"))
    queue = maintenance_queue(scenario_snapshot)
    queue["life_pct"] = (queue["life_pct"] * 100).round(1).astype(str) + "%"
    queue["risk_status"] = queue["risk_status"].map(tr)
    queue["recommended_action"] = queue["recommended_action"].map(tr)
    st.caption(txt(
        "Top 5 shown. Rule-based operational queue: Critical first, then High, then lower predicted/actual RUL.",
        "Top 5 werden angezeigt. Regelbasierte operative Warteschlange: zuerst Kritisch, dann Hoch, danach niedrigere prognostizierte/tatsächliche RUL.",
        "İlk 5 kayıt gösterilir. Kural tabanlı operasyon kuyruğu: önce Kritik, sonra Yüksek, ardından düşük tahmini/gerçek RUL.",
    ))
    st.dataframe(
        queue.rename(
            columns={
                "engine_id": txt("engine_id", "motor_id", "motor_id"),
                "risk_status": txt("risk_status", "risikostatus", "risk_durumu"),
                "predicted_rul": txt("predicted_rul", "prognose_rul", "tahmini_rul"),
                "actual_rul": txt("actual_rul", "tatsächliche_rul", "gerçek_rul"),
                "life_pct": txt("life_consumed", "lebensdauer_verbraucht", "ömür_tüketimi"),
                "recommended_action": txt("recommended_action", "empfohlene_aktion", "önerilen_aksiyon"),
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader(txt("Sensor Behavior", "Sensorverhalten", "Sensör Davranışı"))
    left, right = st.columns([1.15, 1])
    with left:
        st.plotly_chart(line_rul(engine_df, txt(
            f"{selected_engine} · RUL Trend up to {inspection_point} inspection",
            f"{selected_engine} · RUL-Trend bis Prüfpunkt {inspection_point}",
            f"{selected_engine} · {inspection_point} kontrolüne kadar RUL trendi",
        )), width="stretch")
        st.caption(txt(
            "RUL trend is clipped to the selected inspection point so the view behaves like an in-service monitoring snapshot.",
            "Der RUL-Trend wird am gewählten Prüfpunkt abgeschnitten, damit die Ansicht wie ein In-Service-Monitoring-Snapshot wirkt.",
            "RUL trendi seçili kontrol noktasında kesilir; görünüm sahadaki anlık izleme ekranı gibi davranır.",
        ))
    with right:
        st.markdown(txt("#### Sensor Signal Panel", "#### Sensorsignal-Panel", "#### Sensör Sinyal Paneli"))
        sensor_badge_row(sensor_signal)
        st.metric(txt("Deviation score", "Abweichungsscore", "Sapma skoru"), sensor_signal["Deviation score"])
        st.plotly_chart(sensor_chart(engine_df, selected_sensor, selected_engine), width="stretch")
        st.caption(sensor_signal["Dashboard interpretation"])

    st.subheader(txt("Explainability & Root Cause Analysis", "Erklärbarkeit & Ursachenanalyse", "Açıklanabilirlik & Kök Neden Analizi"))
    st.write(txt(
        "This panel explains the selected engine context with thesis artifacts and a simple sensor-shift view. A later version can connect this panel to saved model-level SHAP values.",
        "Dieses Panel erklärt den Kontext des ausgewählten Motors mit Thesis-Artefakten und einer einfachen Sensorverschiebungsansicht. Eine spätere Version kann dieses Panel mit gespeicherten SHAP-Werten auf Modellebene verbinden.",
        "Bu panel seçili motor bağlamını tez çıktıları ve basit sensör kayması görünümüyle açıklar. Sonraki versiyonda bu alan kaydedilmiş model seviyesindeki SHAP değerlerine bağlanabilir.",
    ))
    left, right = st.columns([1, 1])
    with left:
        image_card(FIGURE_DIR / "shap.png", txt("SHAP summary from thesis experiments", "SHAP-Zusammenfassung aus Thesis-Experimenten", "Tez deneylerinden SHAP özeti"))
    with right:
        image_card(FIGURE_DIR / "truevspredicted.png", txt("True vs predicted RUL from saved experiment artifacts", "Tatsächliche vs. prognostizierte RUL aus gespeicherten Experimentartefakten", "Kaydedilmiş deney çıktılarından gerçek ve tahmini RUL"))

    top_sensor_list = explanation["sensor"].head(3).tolist()
    top_sensors = ", ".join(top_sensor_list)
    st.info(
        txt(
            f"{top_sensors} are the strongest degradation-related drivers in this dashboard context. This statement comes from the saved SHAP artifact and selected-engine sensor shift ranking, not from live inference.",
            f"{top_sensors} sind in diesem Dashboard-Kontext die stärksten degradationsbezogenen Treiber. Diese Aussage stammt aus dem gespeicherten SHAP-Artefakt und dem Sensorverschiebungsranking des ausgewählten Motors, nicht aus Live-Inferenz.",
            f"{top_sensors} bu dashboard bağlamında bozulmayla en güçlü ilişkili sürücülerdir. Bu yorum kaydedilmiş SHAP çıktısı ve seçili motorun sensör kayması sıralamasından gelir; canlı model çıkarımı değildir.",
        )
    )
    st.markdown(txt(
        f"#### Top Contributing Sensors · {selected_engine}",
        f"#### Wichtigste beitragende Sensoren · {selected_engine}",
        f"#### En Çok Katkı Veren Sensörler · {selected_engine}",
    ))
    st.plotly_chart(top_sensor_bar(explanation, selected_engine), width="stretch")
    st.dataframe(
        explanation.rename(columns={
            "sensor": txt("sensor", "sensor", "sensör"),
            "absolute_shift": txt("absolute_shift", "absolute_verschiebung", "mutlak_kayma"),
            "dashboard_link": txt("dashboard_link", "dashboard_link", "dashboard_bağlantısı"),
        }),
        width="stretch",
        hide_index=True,
    )

    with st.expander(txt("Recent Snapshot / Raw Data", "Aktueller Snapshot / Rohdaten", "Son Anlık Görünüm / Ham Veri"), expanded=False):
        st.caption(txt(
            "Last observed rows up to the selected inspection point. This table is for traceability, not the primary decision view.",
            "Zuletzt beobachtete Zeilen bis zum gewählten Prüfpunkt. Diese Tabelle dient der Nachvollziehbarkeit, nicht der primären Entscheidungsansicht.",
            "Seçili kontrol noktasına kadar son gözlenen satırlar. Bu tablo karar görünümü için değil, izlenebilirlik içindir.",
        ))
        st.dataframe(
            engine_df.tail(5)[["cycle", "rul", "rul_capped_125", "risk_band", selected_sensor, "op_setting_1", "op_setting_2", "op_setting_3"]],
            width="stretch",
            hide_index=True,
        )

else:
    with st.sidebar:
        st.divider()
        st.header(txt("Bearing Controls", "Lager-Steuerung", "Rulman Kontrolleri"))
        selected_bearing = st.selectbox(txt("Bearing / channel", "Lager / Kanal", "Rulman / kanal"), ["bearing_1", "bearing_2", "bearing_3", "bearing_4"])
        st.divider()
        st.caption(txt(
            "Bearing/channel changes RMS trend, threshold, anomaly count, anomaly table, and recommendation.",
            "Lager/Kanal ändert RMS-Trend, Schwelle, Anomalieanzahl, Anomalietabelle und Empfehlung.",
            "Rulman/kanal seçimi RMS trendini, eşiği, anomali sayısını, anomali tablosunu ve öneriyi değiştirir.",
        ))

    st.info(txt(
        "This mode detects abnormal bearing behavior using RMS vibration signals.",
        "Dieser Modus erkennt abnormales Lagerverhalten anhand von RMS-Vibrationssignalen.",
        "Bu mod RMS titreşim sinyallerini kullanarak anormal rulman davranışını tespit eder.",
    ))

    bearing_rms_col = f"{selected_bearing}_rms"
    bearing_threshold_col = f"{selected_bearing}_threshold"
    bearing_flag_col = f"{selected_bearing}_anomaly_flag"
    selected_bearing_anomalies = bearing[bearing[bearing_flag_col]].copy()
    threshold = float(bearing[bearing_threshold_col].iloc[0])
    max_rms = float(bearing[bearing_rms_col].max())
    first_anomaly = selected_bearing_anomalies["timestamp"].iloc[0] if not selected_bearing_anomalies.empty else None
    latest_anomaly = selected_bearing_anomalies["timestamp"].iloc[-1] if not selected_bearing_anomalies.empty else None
    bearing_context = bearing_operational_context(
        bearing,
        selected_bearing,
        len(selected_bearing_anomalies),
        threshold,
        max_rms,
    )

    st.subheader(txt("Bearing Anomaly Overview", "Lager-Anomalieübersicht", "Rulman Anomali Özeti"))
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi(txt("Selected channel", "Ausgewählter Kanal", "Seçili kanal"), selected_bearing.replace("_", " ").title())
    with c2:
        kpi("Max RMS", f"{max_rms:.3f}")
    with c3:
        kpi(txt("Threshold", "Schwelle", "Eşik"), f"{threshold:.3f}")
    with c4:
        kpi(txt("Anomaly points", "Anomaliepunkte", "Anomali noktası"), f"{len(selected_bearing_anomalies):,}")
    with c5:
        kpi(txt("Anomaly rate", "Anomalierate", "Anomali oranı"), f"{float(bearing_context['anomaly_rate']) * 100:.1f}%")

    st.markdown(txt("#### Bearing Summary", "#### Lager-Zusammenfassung", "#### Rulman Özeti"))
    st.write(bearing_summary_text(selected_bearing, bearing_context, len(selected_bearing_anomalies)))

    st.markdown(txt("#### Bearing Business Impact", "#### Geschäftsauswirkung des Lagers", "#### Rulman İş Etkisi"))
    b1, b2, b3 = st.columns(3)
    with b1:
        value_card(
            txt("Maintenance urgency", "Wartungsdringlichkeit", "Bakım aciliyeti"),
            str(bearing_context["maintenance_urgency"]),
            RISK_COLORS.get(str(bearing_context["maintenance_urgency"]), "#246bfe"),
        )
    with b2:
        value_card(
            txt("Failure progression", "Fehlerfortschritt", "Arıza ilerleyişi"),
            str(bearing_context["progression"]),
            "#d64545" if bearing_context["progression"] == "Severe escalation" else "#f28e2b",
        )
    with b3:
        value_card(
            txt("Recommended action", "Empfohlene Aktion", "Önerilen aksiyon"),
            str(bearing_context["recommended_action"]),
            "#246bfe",
        )
    st.caption(txt(
        f"Basis: {bearing_context['basis']}. Rule-based decision support only; no downtime duration or cost estimate is inferred.",
        f"Basis: {bearing_context['basis']}. Nur regelbasierte Entscheidungsunterstützung; es wird keine Stillstands- oder Kostenschätzung abgeleitet.",
        f"Dayanak: {bearing_context['basis']}. Yalnızca kural tabanlı karar desteğidir; duruş süresi veya maliyet tahmini yapılmaz.",
    ))

    if first_anomaly is not None:
        st.warning(txt(
            f"Maintenance recommendation: anomaly behavior detected from {first_anomaly}. Inspect vibration trend and prepare maintenance review.",
            f"Wartungsempfehlung: Anomalieverhalten ab {first_anomaly} erkannt. Vibrationstrend prüfen und Wartungsbewertung vorbereiten.",
            f"Bakım önerisi: {first_anomaly} itibarıyla anomali davranışı tespit edildi. Titreşim trendini inceleyin ve bakım değerlendirmesi hazırlayın.",
        ))
    else:
        st.success(txt(
            "Maintenance recommendation: no anomaly detected for the selected bearing/channel. Continue routine monitoring.",
            "Wartungsempfehlung: Keine Anomalie für das ausgewählte Lager/den Kanal erkannt. Routineüberwachung fortsetzen.",
            "Bakım önerisi: seçili rulman/kanal için anomali tespit edilmedi. Rutin izlemeye devam edin.",
        ))

    st.plotly_chart(bearing_chart(bearing, selected_bearing), width="stretch")

    st.subheader(txt("Anomaly Summary", "Anomalie-Zusammenfassung", "Anomali Özeti"))
    summary = pd.DataFrame(
        [
            {
                txt("bearing_channel", "lager_kanal", "rulman_kanalı"): selected_bearing,
                txt("samples", "stichproben", "örnek_sayısı"): len(bearing),
                txt("anomaly_points", "anomaliepunkte", "anomali_noktası"): len(selected_bearing_anomalies),
                txt("anomaly_rate", "anomalierate", "anomali_oranı"): f"{float(bearing_context['anomaly_rate']) * 100:.1f}%",
                txt("normal_points", "normalpunkte", "normal_nokta"): len(bearing) - len(selected_bearing_anomalies),
                txt("threshold", "schwelle", "eşik"): round(threshold, 4),
                "max_rms": round(max_rms, 4),
                txt("first_anomaly", "erste_anomalie", "ilk_anomali"): first_anomaly,
                txt("latest_anomaly", "letzte_anomalie", "son_anomali"): latest_anomaly,
                txt("severity", "schweregrad", "şiddet"): tr(bearing_context["severity"]),
                txt("failure_progression", "fehlerfortschritt", "arıza_ilerleyişi"): tr(bearing_context["progression"]),
            }
        ]
    )
    st.dataframe(summary, width="stretch", hide_index=True)

    st.markdown(txt(
        f"#### Anomaly Points · {selected_bearing.replace('_', ' ').title()}",
        f"#### Anomaliepunkte · {selected_bearing.replace('_', ' ').title()}",
        f"#### Anomali Noktaları · {selected_bearing.replace('_', ' ').title()}",
    ))
    anomaly_table = selected_bearing_anomalies[
        ["sample_index", "timestamp", "hours_from_start", bearing_rms_col, bearing_threshold_col, bearing_flag_col]
    ].head(20)
    st.dataframe(anomaly_table, width="stretch", hide_index=True)
