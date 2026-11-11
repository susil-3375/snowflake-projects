import os

import streamlit as st

st.set_page_config(page_title="IT Support Executive Dashboard", layout="wide")
st.title("IT Support Executive Dashboard")
st.caption("ServiceNow incidents · IT_SUPPORT_DB.SERVICENOW")

conn = st.connection("snowflake", ttl=os.getenv("SNOWFLAKE_CONNECTION_TTL"))
T = "IT_SUPPORT_DB.SERVICENOW.INCIDENT_ENRICHED"


@st.cache_data(ttl=600)
def load(sql: str):
    return conn.query(sql)


def clear_cache():
    load.clear()


st.button("Refresh data", on_click=clear_cache)

with st.spinner("Loading..."):
    kpi = load(f"""
        SELECT COUNT_IF(STATUS_GROUP='Open') OPEN_INC,
               ROUND(AVG(RESOLUTION_HOURS),1) MTTR,
               COUNT_IF(STATUS_GROUP='Open' AND PRIORITY<=2) CRIT_OPEN,
               COUNT_IF(STATUS_GROUP='Open' AND DATEDIFF(day,OPENED_AT,CURRENT_TIMESTAMP())>7) AGED,
               COUNT(*) TOTAL
        FROM {T}""").iloc[0]
    backlog = load(f"""
        SELECT AI_CATEGORY CATEGORY, 'P'||PRIORITY PRIORITY, COUNT(*) OPEN_TICKETS
        FROM {T} WHERE STATUS_GROUP='Open' GROUP BY 1,2 ORDER BY 1,2""")
    trend = load(f"""
        WITH o AS (SELECT DATE_TRUNC('week',OPENED_AT) WK, COUNT(*) N FROM {T} GROUP BY 1),
             r AS (SELECT DATE_TRUNC('week',RESOLVED_AT) WK, COUNT(*) N, AVG(RESOLUTION_HOURS) MTTR
                   FROM {T} WHERE RESOLVED_AT IS NOT NULL GROUP BY 1)
        SELECT COALESCE(o.WK,r.WK) WEEK, COALESCE(o.N,0) OPENED, COALESCE(r.N,0) RESOLVED, ROUND(r.MTTR,1) MTTR_HOURS
        FROM o FULL OUTER JOIN r ON o.WK=r.WK ORDER BY 1""")
    mttr_cat = load(f"""
        SELECT AI_CATEGORY CATEGORY, ROUND(AVG(RESOLUTION_HOURS),1) MTTR_HOURS
        FROM {T} GROUP BY 1 ORDER BY 2 DESC""")
    patterns = load("""
        SELECT AI_CATEGORY, ROOT_CAUSE, OCCURRENCES, AVG_RESOLUTION_HOURS, PATTERN_FLAG
        FROM IT_SUPPORT_DB.SERVICENOW.ROOT_CAUSE_PATTERNS ORDER BY OCCURRENCES DESC LIMIT 10""")
    top_open = load(f"""
        SELECT NUMBER, AI_CATEGORY, PRIORITY, PRIORITY_SCORE, STATE, AI_SUMMARY, AI_RECOMMENDATION
        FROM {T} WHERE STATUS_GROUP='Open' ORDER BY PRIORITY_SCORE DESC LIMIT 15""")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Open incidents", int(kpi["OPEN_INC"]), border=True)
c2.metric("MTTR (hours)", float(kpi["MTTR"]), border=True)
c3.metric("Open P1/P2", int(kpi["CRIT_OPEN"]), border=True)
c4.metric("Backlog aged > 7 days", int(kpi["AGED"]), border=True)

left, right = st.columns(2)
with left.container(border=True):
    st.subheader("Ticket backlog by category & priority")
    st.bar_chart(backlog, x="CATEGORY", y="OPEN_TICKETS", color="PRIORITY", stack=True)
with right.container(border=True):
    st.subheader("MTTR by category (hours)")
    st.bar_chart(mttr_cat, x="CATEGORY", y="MTTR_HOURS", horizontal=True)

left, right = st.columns(2)
with left.container(border=True):
    st.subheader("Resolution trend: opened vs resolved per week")
    st.line_chart(trend, x="WEEK", y=["OPENED", "RESOLVED"])
with right.container(border=True):
    st.subheader("Weekly MTTR trend (hours)")
    st.line_chart(trend, x="WEEK", y="MTTR_HOURS")

with st.container(border=True):
    st.subheader("Top recurring root causes")
    st.dataframe(patterns, hide_index=True, use_container_width=True)

with st.container(border=True):
    st.subheader("Highest-priority open incidents (AI-enriched)")
    st.dataframe(
        top_open, hide_index=True, use_container_width=True,
        column_config={"PRIORITY_SCORE": st.column_config.ProgressColumn("Priority score", min_value=0, max_value=100, format="%d")},
    )
