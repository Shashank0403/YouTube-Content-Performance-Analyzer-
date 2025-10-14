"""
Enhanced UI version of the YouTube Content Performance Analyzer
"""

import json
import streamlit as st
from streamlit_echarts import st_echarts
from millify import millify
from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from transform import (
    parse_video,
    youtube_metrics,
    get_video_published_date,
    get_delta_str,
)
import pandas as pd

# --------------------- PAGE CONFIG ---------------------
st.set_page_config(
    page_title="YouTube Content Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------- CUSTOM CSS ---------------------
st.markdown(
    """
    <style>
    /* General App Styling */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #f9fafc 0%, #eef2f7 100%);
    }
    [data-testid="stHeader"] {
        background: rgba(255,255,255,0.5);
    }
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
    }
    .stMetric {
        background-color: white !important;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-container {
        background: white;
        border-radius: 12px;
        padding: 10px 20px;
        box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    }
    .stSubheader {
        color: #2e3b4e;
    }
    .section {
        background: #ffffff;
        border-radius: 15px;
        padding: 25px;
        margin-bottom: 25px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------- TITLE ---------------------
st.markdown(
    "<h1 style='text-align:center; color:#e63946;'>📊 YouTube Content Performance Analyzer</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p style='text-align:center; color:#555;'>Analyze YouTube video performance, engagement metrics, and sentiment insights in one dashboard.</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# --------------------- INPUT ---------------------
with st.container():
    st.markdown("### 🔗 Enter a YouTube Video URL")
    VIDEO_URL = st.text_input(
        "Paste the YouTube video URL below:",
        placeholder="https://www.youtube.com/watch?v=example",
        label_visibility="collapsed",
    )

# --------------------- MAIN CONTENT ---------------------
try:
    if VIDEO_URL:
        with st.spinner("🔍 Analyzing video data... Please wait..."):
            df = parse_video(VIDEO_URL)
            df_metrics = youtube_metrics(VIDEO_URL)

            # ---------- Metrics Section ----------
            with st.container():
                st.markdown("## 📈 Key Metrics")
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                col1.metric("Views", millify(df_metrics[0], precision=2))
                col2.metric("Likes", millify(df_metrics[1], precision=2))
                col3.metric("Comments", millify(df_metrics[2], precision=2))
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")

            # ---------- Video Preview ----------
            st.subheader("🎥 Video Preview")
            st.video(VIDEO_URL)

            # ---------- Published Info ----------
            @st.fragment
            def tz_choice_section():
                df_published_date = get_video_published_date(VIDEO_URL)
                delta_str = get_delta_str(df_published_date)
                title = df_published_date.get("Title", "")
                creator = df_published_date.get("Creator", "")

                with st.container():
                    st.markdown("## 🕒 Video Details")
                    st.markdown("<div class='section'>", unsafe_allow_html=True)
                    if title:
                        st.markdown(f"### {title}")
                    if creator:
                        st.markdown(f"**Creator:** {creator}")

                    tz_choice = st.segmented_control(
                        "Published",
                        label_visibility="hidden",
                        options=["UTC", "EST", "IST"],
                        default="UTC",
                        selection_mode="single",
                    )

                    st.metric(
                        f"Published ({tz_choice})",
                        df_published_date[tz_choice],
                        delta=delta_str,
                    )
                    st.markdown("</div>", unsafe_allow_html=True)

            tz_choice_section()

            # ---------- Top Comments ----------
            with st.container():
                st.markdown("## 💬 Most Liked Comments")
                df_top = (
                    df[["Author", "Comment", "Timestamp", "Likes"]]
                    .sort_values("Likes", ascending=False)
                    .reset_index(drop=True)
                )
                top_11 = df_top.head(11)
                gd1 = GridOptionsBuilder.from_dataframe(top_11)
                gd1.configure_auto_height(True)
                gridoptions1 = gd1.build()

                AgGrid(
                    top_11,
                    gridOptions=gridoptions1,
                    key="top_comments",
                    theme="streamlit",
                    update_on="MANUAL",
                )

            # ---------- Languages ----------
            with st.container():
                st.markdown("## 🌍 Languages Used")
                df_langs = (
                    df["Language"]
                    .value_counts()
                    .rename_axis("Language")
                    .reset_index(name="count")
                )

                options2 = {
                    "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                    "yAxis": {"type": "category", "data": df_langs["Language"].tolist()},
                    "xAxis": {"type": "value"},
                    "series": [{"data": df_langs["count"].tolist(), "type": "bar"}],
                    "color": ["#3b82f6"],
                }
                st_echarts(options=options2, height="450px")

            # ---------- Most Replied Comments ----------
            with st.container():
                st.markdown("## 🔁 Most Replied Comments")
                df_replies = (
                    df[["Author", "Comment", "Timestamp", "TotalReplies"]]
                    .sort_values("TotalReplies", ascending=False)
                    .reset_index(drop=True)
                )
                gd2 = GridOptionsBuilder.from_dataframe(df_replies.head(11))
                gd2.configure_auto_height(True)
                gridoptions2 = gd2.build()
                AgGrid(
                    df_replies.head(11),
                    gridOptions=gridoptions2,
                    key="top_replies",
                    theme="streamlit",
                    update_on="MANUAL",
                )

            # ---------- Sentiment Analysis ----------
            with st.container():
                st.markdown("## ❤️ Comment Sentiment Analysis")
                sentiments = df[df["Language"] == "English"]
                data_sentiments = (
                    sentiments["TextBlob_Sentiment_Type"]
                    .value_counts()
                    .rename_axis("Sentiment")
                    .reset_index(name="counts")
                )

                data_sentiments["Review_percent"] = (
                    100.0 * data_sentiments["counts"] / data_sentiments["counts"].sum()
                ).round(1)

                if "No sentiment data" in data_sentiments["Sentiment"].values:
                    data_list = [
                        {
                            "value": int(data_sentiments["counts"].iloc[0]),
                            "name": "No sentiment data",
                        }
                    ]
                else:
                    percent_map = {
                        row["Sentiment"]: float(row["Review_percent"])
                        for _, row in data_sentiments.iterrows()
                    }
                    data_list = [
                        {"value": percent_map.get("NEUTRAL", 0.0), "name": "NEUTRAL"},
                        {"value": percent_map.get("POSITIVE", 0.0), "name": "POSITIVE"},
                        {"value": percent_map.get("NEGATIVE", 0.0), "name": "NEGATIVE"},
                    ]

                options = {
                    "tooltip": {"trigger": "item", "formatter": "{d}%"},
                    "legend": {"top": "5%", "left": "center"},
                    "series": [
                        {
                            "name": "Sentiment",
                            "type": "pie",
                            "radius": ["40%", "70%"],
                            "avoidLabelOverlap": False,
                            "itemStyle": {
                                "borderRadius": 10,
                                "borderColor": "#fff",
                                "borderWidth": 2,
                            },
                            "label": {"show": False, "position": "center"},
                            "emphasis": {
                                "label": {
                                    "show": True,
                                    "fontSize": "30",
                                    "fontWeight": "bold",
                                }
                            },
                            "labelLine": {"show": False},
                            "data": data_list,
                        }
                    ],
                    "color": ["#f87171", "#34d399", "#60a5fa"],
                }
                st_echarts(options=options, height="450px")

except Exception as e:
    st.error(e, icon="🚨")
