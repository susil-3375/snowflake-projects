"""CODEBLUE_COMMAND_CENTER - Code Blue risk, clinical monitoring and ICU capacity planning.

Data source: semantic view DEMO_DB.CLINIC.CODEBLUE_PREDICTION_SV
(base table DEMO_DB.CLINIC.CODEBLUE_PREDICTION_RESULTS).
"""

import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="CodeBlue Command Center", page_icon=":material/emergency:", layout="wide")

RISK_ORDER = ["High", "Medium", "Low"]
RISK_COLORS = alt.Scale(domain=RISK_ORDER, range=["#C0392B", "#F39C12", "#27AE60"])

# One row per patient: semantic-view dimensions + per-patient metrics (the SV cannot mix
# FACTS with DIMENSIONS). TEMPERATURE has no SV metric, so it is joined from the base table.
PATIENT_SQL = """
SELECT sv.PRIMARY_MRN            AS PRIMARYMRN,
       sv.AVG_AGE::INT           AS AGE,
       sv.GENDER,
       sv.ENTRY_DATE,
       sv.DISCHARGE_DATE,
       sv.RISK_BAND,
       sv.AGE_GROUP,
       sv.TOP_RISK_DRIVER_1,
       sv.PREDICTED_CODEBLUE_COUNT AS CODEBLUE_PREDICTION,
       sv.AVG_CODEBLUE_PROBABILITY AS CODEBLUE_PROBABILITY,
       sv.AVG_RESPIRATION        AS RESPIRATION,
       sv.AVG_PULSE              AS PULSE,
       sv.AVG_SYSTOLIC_BP        AS SYSTOLIC_BP,
       b.TEMPERATURE,
       sv.AVG_PULSE_OXIMETRY     AS PULSE_OXIMETRY,
       sv.AVG_BNP                AS BNP,
       sv.AVG_ALBUMIN            AS ALBUMIN,
       sv.AVG_LENGTH_OF_STAY     AS LOS_DAYS
FROM SEMANTIC_VIEW(
        DEMO_DB.CLINIC.CODEBLUE_PREDICTION_SV
        DIMENSIONS predictions.primary_mrn, predictions.gender, predictions.entry_date,
                   predictions.discharge_date, predictions.risk_band, predictions.age_group,
                   predictions.top_risk_driver_1
        METRICS predictions.avg_age, predictions.avg_codeblue_probability,
                predictions.predicted_codeblue_count, predictions.avg_respiration,
                predictions.avg_pulse, predictions.avg_systolic_bp, predictions.avg_pulse_oximetry,
                predictions.avg_bnp, predictions.avg_albumin, predictions.avg_length_of_stay
     ) sv
LEFT JOIN DEMO_DB.CLINIC.CODEBLUE_PREDICTION_RESULTS b
       ON b.PRIMARYMRN = sv.PRIMARY_MRN
"""

NUMERIC_COLS = ["AGE", "CODEBLUE_PREDICTION", "CODEBLUE_PROBABILITY", "RESPIRATION", "PULSE",
                "SYSTOLIC_BP", "TEMPERATURE", "PULSE_OXIMETRY", "BNP", "ALBUMIN", "LOS_DAYS"]


@st.cache_data(ttl=600, show_spinner=False)
def load_patients() -> pd.DataFrame:
    conn = st.connection("snowflake", ttl=os.getenv("SNOWFLAKE_CONNECTION_TTL"))
    df = conn.query(PATIENT_SQL, ttl=0)
    df.columns = [c.upper() for c in df.columns]
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ENTRY_DATE"] = pd.to_datetime(df["ENTRY_DATE"])
    df["DISCHARGE_DATE"] = pd.to_datetime(df["DISCHARGE_DATE"])
    df["ENTRY_DAY"] = df["ENTRY_DATE"].dt.normalize()
    return df


def refresh_data() -> None:
    load_patients.clear()


def linear_forecast(series: pd.Series, horizon: int, lookback: int) -> pd.DataFrame:
    """Fit a straight line to the last `lookback` days and project `horizon` days ahead."""
    hist = series.tail(lookback)
    x = np.arange(len(hist))
    slope, intercept = np.polyfit(x, hist.values.astype(float), 1) if len(hist) > 1 else (0.0, float(hist.mean()))
    resid_sd = float(np.std(hist.values - (slope * x + intercept))) if len(hist) > 2 else 0.0
    fx = np.arange(len(hist), len(hist) + horizon)
    yhat = np.clip(slope * fx + intercept, 0, None)
    dates = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    return pd.DataFrame({"DATE": dates, "FORECAST": yhat,
                         "LOWER": np.clip(yhat - 1.96 * resid_sd, 0, None),
                         "UPPER": yhat + 1.96 * resid_sd})


def daily_occupancy(icu: pd.DataFrame, days: pd.DatetimeIndex) -> pd.Series:
    """Patients in ICU on each day: admitted on/before the day and discharged after it."""
    entry = icu["ENTRY_DAY"].values
    disch = icu["DISCHARGE_DATE"].dt.normalize().values
    return pd.Series([int(((entry <= d) & (disch > d)).sum()) for d in days.values], index=days)


# ---------------------------------------------------------------- header + sidebar
st.title(":material/emergency: CodeBlue Command Center")
st.caption("Real-time Code Blue risk, patient deterioration and ICU capacity • Source: DEMO_DB.CLINIC.CODEBLUE_PREDICTION_SV")

with st.sidebar:
    st.header("Settings")
    st.button("Refresh data", icon=":material/refresh:", on_click=refresh_data, use_container_width=True)
    st.subheader("ICU assumptions")
    icu_capacity = st.number_input("ICU bed capacity", min_value=1, max_value=1000, value=50, step=1)
    icu_basis = st.radio("Predicted ICU admission =", ["Model predicts Code Blue", "High risk band (≥ 70%)"])
    target_util = st.slider("Target utilization %", 50, 100, 85)
    horizon = st.slider("Forecast horizon (days)", 7, 30, 14)
    lookback = st.slider("Trend lookback (days)", 7, 60, 21)
    st.subheader("Immediate intervention criteria")
    st.caption("High risk AND any of:")
    sbp_lo = st.number_input("Systolic BP <", value=90)
    hr_hi = st.number_input("Pulse >", value=120)
    rr_hi = st.number_input("Respiration >", value=24)
    spo2_lo = st.number_input("SpO2 <", value=90)
    st.caption("Risk bands follow the semantic view: High ≥ 0.7, Medium > 0.3, Low ≤ 0.3.")

with st.spinner("Loading patient predictions..."):
    try:
        df = load_patients()
    except Exception as e:  # surface Snowflake errors clearly
        st.error(f"Could not load data from CODEBLUE_PREDICTION_SV: {e}")
        st.stop()

if df.empty:
    st.warning("No prediction rows found.")
    st.stop()

# ---------------------------------------------------------------- derived fields
df["ABNORMAL_VITALS"] = ((df["SYSTOLIC_BP"] < sbp_lo) | (df["PULSE"] > hr_hi)
                         | (df["RESPIRATION"] > rr_hi) | (df["PULSE_OXIMETRY"] < spo2_lo))
df["IMMEDIATE"] = (df["RISK_BAND"] == "High") & df["ABNORMAL_VITALS"]
df["ICU_FLAG"] = (df["CODEBLUE_PREDICTION"] >= 1) if icu_basis.startswith("Model") else (df["RISK_BAND"] == "High")

as_of = df["ENTRY_DAY"].max()
all_days = pd.date_range(df["ENTRY_DAY"].min(), as_of, freq="D")
icu = df[df["ICU_FLAG"]]
occupancy = daily_occupancy(icu, all_days)
daily = (df.groupby("ENTRY_DAY")
           .agg(PATIENTS=("PRIMARYMRN", "count"),
                HIGH_RISK=("RISK_BAND", lambda s: int((s == "High").sum())),
                AVG_PROB=("CODEBLUE_PROBABILITY", "mean"),
                ICU_ADMITS=("ICU_FLAG", "sum"))
           .reindex(all_days, fill_value=0))
daily.index.name = "DATE"
daily["AVG_PROB"] = daily["AVG_PROB"].replace(0, np.nan)
daily_reset = daily.reset_index()

counts = df["RISK_BAND"].value_counts()
n_total = len(df)
current_occ = int(occupancy.iloc[-1]) if len(occupancy) else 0

tab_exec, tab_clin, tab_icu = st.tabs([":material/insights: Executive Summary",
                                       ":material/monitor_heart: Clinical Risk Monitoring",
                                       ":material/bed: ICU Capacity Planning"])

# ================================================================ 1. EXECUTIVE SUMMARY
with tab_exec:
    st.subheader("Hospital risk & readiness at a glance")
    st.caption(f"Data through {as_of:%b %d, %Y}")
    r1 = st.columns(4)
    r1[0].metric("Total Patients Monitored", f"{n_total:,}", border=True)
    r1[1].metric("High-Risk Patients", f"{counts.get('High', 0):,}",
                 f"{counts.get('High', 0) / n_total:.1%} of total", delta_color="off", border=True)
    r1[2].metric("Medium-Risk Patients", f"{counts.get('Medium', 0):,}",
                 f"{counts.get('Medium', 0) / n_total:.1%} of total", delta_color="off", border=True)
    r1[3].metric("Low-Risk Patients", f"{counts.get('Low', 0):,}",
                 f"{counts.get('Low', 0) / n_total:.1%} of total", delta_color="off", border=True)
    r2 = st.columns(4)
    r2[0].metric("Avg Prediction Probability", f"{df['CODEBLUE_PROBABILITY'].mean():.1%}", border=True)
    r2[1].metric("Predicted ICU Admissions", f"{int(df['ICU_FLAG'].sum()):,}", border=True)
    r2[2].metric("ICU Bed Utilization", f"{current_occ / icu_capacity:.0%}",
                 f"{current_occ} of {icu_capacity} beds", delta_color="off", border=True)
    r2[3].metric("Immediate Intervention", f"{int(df['IMMEDIATE'].sum()):,}",
                 "High risk + abnormal vitals", delta_color="off", border=True)

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("**Risk Distribution**")
        rd = counts.reindex(RISK_ORDER, fill_value=0).rename_axis("RISK_BAND").reset_index(name="PATIENTS")
        st.altair_chart(
            alt.Chart(rd).mark_arc(innerRadius=60).encode(
                theta="PATIENTS:Q",
                color=alt.Color("RISK_BAND:N", scale=RISK_COLORS, sort=RISK_ORDER, title="Risk band"),
                tooltip=["RISK_BAND", "PATIENTS"]),
            use_container_width=True)
    with c2.container(border=True):
        st.markdown("**Prediction Probability Distribution**")
        st.altair_chart(
            alt.Chart(df).mark_bar().encode(
                x=alt.X("CODEBLUE_PROBABILITY:Q", bin=alt.Bin(step=0.05), title="Code Blue probability",
                        axis=alt.Axis(format="%")),
                y=alt.Y("count():Q", title="Patients"),
                color=alt.Color("RISK_BAND:N", scale=RISK_COLORS, sort=RISK_ORDER, title="Risk band"),
                tooltip=[alt.Tooltip("count():Q", title="Patients")]),
            use_container_width=True)

    with st.container(border=True):
        st.markdown("**Patient Risk Trend** (weekly)")
        weekly = (df.set_index("ENTRY_DATE")
                    .groupby([pd.Grouper(freq="W-MON", label="left", closed="left"), "RISK_BAND"])
                    .size().rename("PATIENTS").reset_index())
        weekly_prob = (df.set_index("ENTRY_DATE")["CODEBLUE_PROBABILITY"]
                         .resample("W-MON", label="left", closed="left").mean().rename("AVG_PROB").reset_index())
        bars = alt.Chart(weekly).mark_bar(opacity=0.85).encode(
            x=alt.X("ENTRY_DATE:T", title="Week"),
            y=alt.Y("PATIENTS:Q", title="Patients admitted"),
            color=alt.Color("RISK_BAND:N", scale=RISK_COLORS, sort=RISK_ORDER, title="Risk band"),
            order=alt.Order("RISK_BAND:N", sort="descending"),
            tooltip=["ENTRY_DATE:T", "RISK_BAND", "PATIENTS"])
        line = alt.Chart(weekly_prob).mark_line(point=True, color="#2C3E50").encode(
            x="ENTRY_DATE:T",
            y=alt.Y("AVG_PROB:Q", title="Avg probability", axis=alt.Axis(format="%")),
            tooltip=[alt.Tooltip("AVG_PROB:Q", format=".1%")])
        st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent"), use_container_width=True)

# ================================================================ 2. CLINICAL RISK MONITORING
with tab_clin:
    st.subheader("Patients requiring immediate attention")
    f1, f2, f3 = st.columns([2, 2, 1])
    bands = f1.multiselect("Risk band", RISK_ORDER, default=["High"])
    genders = f2.multiselect("Gender", sorted(df["GENDER"].dropna().unique()),
                             default=sorted(df["GENDER"].dropna().unique()))
    only_immediate = f3.toggle("Immediate only", value=False)

    view = df[df["RISK_BAND"].isin(bands) & df["GENDER"].isin(genders)]
    if only_immediate:
        view = view[view["IMMEDIATE"]]
    view = view.sort_values("CODEBLUE_PROBABILITY", ascending=False)

    list_cols = ["PRIMARYMRN", "AGE", "GENDER", "ENTRY_DATE", "CODEBLUE_PREDICTION", "CODEBLUE_PROBABILITY",
                 "RESPIRATION", "PULSE", "SYSTOLIC_BP", "TEMPERATURE", "BNP", "ALBUMIN"]
    with st.container(border=True):
        st.markdown(f"**High-Risk Patient List** — {len(view):,} patients")
        st.dataframe(
            view[list_cols], hide_index=True, use_container_width=True, height=380,
            column_config={
                "PRIMARYMRN": "MRN",
                "ENTRY_DATE": st.column_config.DatetimeColumn("Entry date", format="YYYY-MM-DD HH:mm"),
                "CODEBLUE_PREDICTION": st.column_config.CheckboxColumn("Predicted Code Blue"),
                "CODEBLUE_PROBABILITY": st.column_config.ProgressColumn(
                    "Probability", min_value=0.0, max_value=1.0, format="percent"),
                "SYSTOLIC_BP": st.column_config.NumberColumn("Systolic BP", format="%d"),
                "TEMPERATURE": st.column_config.NumberColumn("Temp (°C)", format="%.1f"),
                "BNP": st.column_config.NumberColumn("BNP", format="%d"),
                "ALBUMIN": st.column_config.NumberColumn("Albumin", format="%.1f"),
            })

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("**Top 20 Highest-Risk Patients**")
        top20 = df.nlargest(20, "CODEBLUE_PROBABILITY")
        st.altair_chart(
            alt.Chart(top20).mark_bar().encode(
                x=alt.X("CODEBLUE_PROBABILITY:Q", title="Code Blue probability",
                        axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("PRIMARYMRN:N", sort="-x", title=None),
                color=alt.Color("CODEBLUE_PROBABILITY:Q", scale=alt.Scale(scheme="reds"), legend=None),
                tooltip=["PRIMARYMRN", "AGE", "GENDER", "TOP_RISK_DRIVER_1",
                         alt.Tooltip("CODEBLUE_PROBABILITY:Q", format=".1%")]).properties(height=480),
            use_container_width=True)
    with c2.container(border=True):
        st.markdown("**Patient Risk Heatmap** — avg probability by age group × top risk driver")
        hm = (df.groupby(["AGE_GROUP", "TOP_RISK_DRIVER_1"])
                .agg(AVG_PROB=("CODEBLUE_PROBABILITY", "mean"), PATIENTS=("PRIMARYMRN", "count"))
                .reset_index())
        base = alt.Chart(hm).encode(
            x=alt.X("TOP_RISK_DRIVER_1:N", title="Top risk driver"),
            y=alt.Y("AGE_GROUP:N", title="Age group", sort=["18-39", "40-64", "65-79", "80+"]))
        heat = base.mark_rect().encode(
            color=alt.Color("AVG_PROB:Q", scale=alt.Scale(scheme="orangered"), title="Avg prob",
                            legend=alt.Legend(format="%")),
            tooltip=["AGE_GROUP", "TOP_RISK_DRIVER_1", "PATIENTS", alt.Tooltip("AVG_PROB:Q", format=".1%")])
        text = base.mark_text(fontSize=10).encode(text="PATIENTS:Q")
        st.altair_chart((heat + text).properties(height=480), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3.container(border=True):
        st.markdown("**Risk Category Breakdown** — by age group and gender")
        rb = df.groupby(["AGE_GROUP", "GENDER", "RISK_BAND"]).size().rename("PATIENTS").reset_index()
        st.altair_chart(
            alt.Chart(rb).mark_bar().encode(
                x=alt.X("AGE_GROUP:N", title="Age group"),
                y=alt.Y("PATIENTS:Q", stack="normalize", title="Share of patients", axis=alt.Axis(format="%")),
                color=alt.Color("RISK_BAND:N", scale=RISK_COLORS, sort=RISK_ORDER, title="Risk band"),
                column=alt.Column("GENDER:N", title=None),
                tooltip=["AGE_GROUP", "GENDER", "RISK_BAND", "PATIENTS"]).properties(width=170),
            use_container_width=False)
    with c4.container(border=True):
        st.markdown("**High-Risk Admissions Trend** (daily + 7-day avg)")
        trend = daily_reset[["DATE", "HIGH_RISK"]].copy()
        trend["7D_AVG"] = trend["HIGH_RISK"].rolling(7, min_periods=1).mean()
        b = alt.Chart(trend).encode(x=alt.X("DATE:T", title="Admission date"))
        st.altair_chart(
            b.mark_bar(color="#F5B7B1").encode(y=alt.Y("HIGH_RISK:Q", title="High-risk admissions"),
                                               tooltip=["DATE:T", "HIGH_RISK"])
            + b.mark_line(color="#C0392B", strokeWidth=2.5).encode(
                y="7D_AVG:Q", tooltip=[alt.Tooltip("7D_AVG:Q", format=".1f")]),
            use_container_width=True)

# ================================================================ 3. ICU CAPACITY PLANNING
with tab_icu:
    st.subheader("ICU demand & capacity")
    st.info(
        f"ICU metrics are **derived**: the source has no ICU census. A patient counts as an ICU admission when "
        f"*{icu_basis.lower()}*, and occupies a bed from entry until discharge. Capacity is the sidebar value "
        f"({icu_capacity} beds). Forecasts use a {lookback}-day linear trend.", icon=":material/info:")

    peak_occ = int(occupancy.max()) if len(occupancy) else 0
    avg_los = icu["LOS_DAYS"].mean() if len(icu) else 0.0
    k = st.columns(6)
    k[0].metric("Current ICU Capacity", f"{icu_capacity}", border=True)
    k[1].metric("Available ICU Beds", f"{max(icu_capacity - current_occ, 0)}", border=True)
    k[2].metric("ICU Utilization Rate", f"{current_occ / icu_capacity:.0%}",
                f"target {target_util}%", delta_color="off", border=True)
    k[3].metric("Peak Occupancy", f"{peak_occ}", f"{occupancy.idxmax():%b %d}" if peak_occ else None,
                delta_color="off", border=True)
    k[4].metric("Avg ICU Length of Stay", f"{avg_los:.1f} days", border=True)
    k[5].metric("Current Occupancy", f"{current_occ}", f"as of {as_of:%b %d}", delta_color="off", border=True)

    demand_fc = linear_forecast(daily["ICU_ADMITS"].astype(float), horizon, lookback)
    occ_fc = linear_forecast(occupancy.astype(float), horizon, lookback)

    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.markdown("**Predicted ICU Demand Trend** — daily ICU admissions with forecast")
        hist = daily_reset[["DATE", "ICU_ADMITS"]]
        h = alt.Chart(hist).mark_line(point=True, color="#2C3E50").encode(
            x=alt.X("DATE:T", title=None), y=alt.Y("ICU_ADMITS:Q", title="ICU admissions / day"),
            tooltip=["DATE:T", "ICU_ADMITS"])
        band = alt.Chart(demand_fc).mark_area(opacity=0.2, color="#C0392B").encode(
            x="DATE:T", y="LOWER:Q", y2="UPPER:Q")
        fl = alt.Chart(demand_fc).mark_line(strokeDash=[6, 4], color="#C0392B").encode(
            x="DATE:T", y="FORECAST:Q", tooltip=["DATE:T", alt.Tooltip("FORECAST:Q", format=".1f")])
        st.altair_chart(h + band + fl, use_container_width=True)
        st.caption(f"Projected ICU admissions next {horizon} days: **{demand_fc['FORECAST'].sum():.0f}**")
    with c2.container(border=True):
        st.markdown("**Daily High-Risk Patient Volume**")
        st.altair_chart(
            alt.Chart(daily_reset).mark_line(point=True, color="#C0392B").encode(
                x=alt.X("DATE:T", title=None), y=alt.Y("HIGH_RISK:Q", title="High-risk patients admitted"),
                tooltip=["DATE:T", "HIGH_RISK", "PATIENTS"]),
            use_container_width=True)

    c3, c4 = st.columns(2)
    with c3.container(border=True):
        st.markdown("**ICU Utilization Forecast**")
        util_hist = pd.DataFrame({"DATE": occupancy.index, "UTIL": occupancy.values / icu_capacity, "TYPE": "Actual"})
        util_fc = pd.DataFrame({"DATE": occ_fc["DATE"], "UTIL": occ_fc["FORECAST"] / icu_capacity, "TYPE": "Forecast"})
        util = pd.concat([util_hist, util_fc])
        lines = alt.Chart(util).mark_line().encode(
            x=alt.X("DATE:T", title=None),
            y=alt.Y("UTIL:Q", title="Utilization", axis=alt.Axis(format="%")),
            color=alt.Color("TYPE:N", scale=alt.Scale(domain=["Actual", "Forecast"], range=["#2C3E50", "#C0392B"]),
                            title=None),
            strokeDash=alt.StrokeDash("TYPE:N", scale=alt.Scale(domain=["Actual", "Forecast"],
                                                                 range=[[1, 0], [6, 4]]), legend=None),
            tooltip=["DATE:T", "TYPE", alt.Tooltip("UTIL:Q", format=".0%")])
        rule = alt.Chart(pd.DataFrame({"y": [target_util / 100]})).mark_rule(color="#F39C12", strokeDash=[3, 3]).encode(y="y:Q")
        st.altair_chart(lines + rule, use_container_width=True)
        peak_fc = util_fc["UTIL"].max() if len(util_fc) else 0
        msg = f"Peak forecast utilization: **{peak_fc:.0%}** (target {target_util}%)."
        (st.warning if peak_fc * 100 > target_util else st.success)(msg)
    with c4.container(border=True):
        st.markdown("**ICU Capacity vs Demand**")
        cvd = pd.concat([
            pd.DataFrame({"DATE": occupancy.index, "BEDS": occupancy.values, "SERIES": "Occupied (actual)"}),
            pd.DataFrame({"DATE": occ_fc["DATE"], "BEDS": occ_fc["FORECAST"], "SERIES": "Occupied (forecast)"}),
        ])
        cap = pd.DataFrame({"DATE": pd.concat([pd.Series(occupancy.index), occ_fc["DATE"]]),
                            "BEDS": icu_capacity, "SERIES": "Capacity"})
        cvd_all = pd.concat([cvd, cap])
        st.altair_chart(
            alt.Chart(cvd_all).mark_line(strokeWidth=2).encode(
                x=alt.X("DATE:T", title=None), y=alt.Y("BEDS:Q", title="Beds"),
                color=alt.Color("SERIES:N", title=None, scale=alt.Scale(
                    domain=["Capacity", "Occupied (actual)", "Occupied (forecast)"],
                    range=["#27AE60", "#2C3E50", "#C0392B"])),
                tooltip=["DATE:T", "SERIES", alt.Tooltip("BEDS:Q", format=".1f")]),
            use_container_width=True)
