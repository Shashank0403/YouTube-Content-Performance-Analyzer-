"""
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
from streamlit_echarts import st_echarts

# ------------------------------
# CONFIG
# ------------------------------
st.set_page_config(page_title="YouTube Content Analyzer", layout="wide")
st.title("📊 YouTube Content Analyzer")

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
analyzer = SentimentIntensityAnalyzer()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"[^A-Za-z0-9\s❤️‍🔥]", "", text)  # remove punctuation but keep hearts/fire
    return text.strip().lower()

def get_vader_sentiment(text):
    """
    Returns sentiment label based on VADER compound score.
    Positive, Neutral, Negative
    """
    score = analyzer.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def get_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

def fetch_video_details(video_id, api_key):
    youtube = build("youtube", "v3", developerKey=api_key)
    req = youtube.videos().list(part="snippet,statistics", id=video_id)
    res = req.execute()
    if not res["items"]:
        return None
    item = res["items"][0]
    return {
        "title": item["snippet"]["title"],
        "channel": item["snippet"]["channelTitle"],
        "published": item["snippet"]["publishedAt"],
        "views": int(item["statistics"].get("viewCount", 0)),
        "likes": int(item["statistics"].get("likeCount", 0)),
        "comments": int(item["statistics"].get("commentCount", 0))
    }

def fetch_comments(video_id, api_key):
    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    next_page_token = None
    while True:
        req = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token
        )
        res = req.execute()
        for item in res["items"]:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "Author": snippet["authorDisplayName"],
                "Text": snippet["textDisplay"],
                "Likes": snippet["likeCount"],
                "PublishedAt": snippet["publishedAt"]
            })
        next_page_token = res.get("nextPageToken")
        if not next_page_token:
            break
    return pd.DataFrame(comments)

# ------------------------------
# MAIN APP
# ------------------------------
api_key = st.secrets["api_key"]
url = st.text_input("Enter YouTube video URL:")

if url:
    video_id = get_video_id(url)
    if not video_id:
        st.error("⚠️ Invalid YouTube URL")
    else:
        st.info("Fetching video details and comments...")
        details = fetch_video_details(video_id, api_key)
        df = fetch_comments(video_id, api_key)

        if details and not df.empty:
            # ------------------------------
            # VIDEO METRICS
            # ------------------------------
            st.subheader("🎥 Video Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Views", f"{details['views']:,}")
            with col2:
                st.metric("Likes", f"{details['likes']:,}")
            with col3:
                st.metric("Comments", f"{details['comments']:,}")
            with col4:
                engagement = round((details["likes"] + details["comments"]) / details["views"] * 100, 2)
                st.metric("Engagement Rate", f"{engagement}%")

            st.write(f"**Title:** {details['title']}")
            st.write(f"**Channel:** {details['channel']}")
            st.write(f"**Published on:** {details['published'][:10]}")

            # ------------------------------
            # COMMENT CLEANING & SENTIMENT
            # ------------------------------
            df["Cleaned_Text"] = df["Text"].apply(clean_text)
            df["Sentiment"] = df["Cleaned_Text"].apply(get_vader_sentiment)
            df["PublishedAt"] = pd.to_datetime(df["PublishedAt"])

            # ------------------------------
            # DOWNLOAD CSV
            # ------------------------------
            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download Comments CSV", csv, "youtube_comments.csv", "text/csv")

            # ------------------------------
            # COMMENT ACTIVITY OVER MONTHS
            # ------------------------------
            st.subheader("📈 Comment Activity Over Time (Monthly)")
            df["MonthYear"] = df["PublishedAt"].dt.to_period("M")
            monthly_activity = df.groupby("MonthYear").size().reset_index(name="Count")
            monthly_activity["MonthYearStr"] = monthly_activity["MonthYear"].astype(str)

            options_month = {
                "tooltip": {"trigger": "axis"},
                "xAxis": {"type": "category", "data": monthly_activity["MonthYearStr"].tolist()},
                "yAxis": {"type": "value"},
                "series": [{"data": monthly_activity["Count"].tolist(), "type": "line", "smooth": True}],
            }
            st_echarts(options=options_month, height="400px")

            # ------------------------------
            # WORD CLOUD
            # ------------------------------
            st.subheader("☁️ Word Cloud of Most Frequent Words")
            all_text = " ".join(df["Cleaned_Text"])
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(all_text)
            fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
            ax_wc.imshow(wordcloud, interpolation='bilinear')
            ax_wc.axis("off")
            st.pyplot(fig_wc)

            # ------------------------------
            # TOP POSITIVE & NEGATIVE COMMENTS
            # ------------------------------
            st.subheader("💬 Sentiment Highlights")
            pos_comments = df[df["Sentiment"] == "Positive"].sort_values("Likes", ascending=False).head(5)
            neg_comments = df[df["Sentiment"] == "Negative"].sort_values("Likes", ascending=False).head(5)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top Positive Comments:** 🌞")
                for _, row in pos_comments.iterrows():
                    st.write(f"👉 {row['Text']}  \n❤️ Likes: {row['Likes']}")
            with col2:
                st.markdown("**Top Negative Comments:** ⚡")
                for _, row in neg_comments.iterrows():
                    st.write(f"👎 {row['Text']}  \n💔 Likes: {row['Likes']}")

            # ------------------------------
            # INTERACTIVE SENTIMENT PIE CHART
            # ------------------------------
            st.subheader("📊 Sentiment Distribution (Interactive)")
            sentiment_counts = df["Sentiment"].value_counts()
            data_list = [
                {"value": int(sentiment_counts.get("Positive", 0)), "name": "Positive"},
                {"value": int(sentiment_counts.get("Neutral", 0)), "name": "Neutral"},
                {"value": int(sentiment_counts.get("Negative", 0)), "name": "Negative"},
            ]

            options = {
                "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
                "legend": {"top": "5%", "left": "center"},
                "series": [
                    {
                        "name": "Sentiment",
                        "type": "pie",
                        "radius": ["40%", "70%"],
                        "avoidLabelOverlap": False,
                        "itemStyle": {"borderRadius": 10, "borderColor": "#fff", "borderWidth": 2},
                        "emphasis": {"label": {"show": True, "fontSize": 20, "fontWeight": "bold"}},
                        "label": {"show": False},
                        "labelLine": {"show": False},
                        "data": data_list,
                    }
                ],
            }
            st_echarts(options=options, height="400px")

        else:
            st.warning("No comments found or video details unavailable.")
