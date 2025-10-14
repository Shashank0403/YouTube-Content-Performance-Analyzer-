import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from googleapiclient.discovery import build
from datetime import datetime
import numpy as np
import re

# ------------------------------
# CONFIGURATION
# ------------------------------
st.set_page_config(page_title="YouTube Content Analyzer", layout="wide")

st.title("📊 YouTube Content Analyzer — Dashboard")

api_key = st.secrets["api_key"]  # ensure your Streamlit Cloud has this secret
youtube = build("youtube", "v3", developerKey=api_key)

# ------------------------------
# HELPER FUNCTIONS
# ------------------------------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"[^A-Za-z0-9\s]", "", text)  # remove punctuation
    return text.strip().lower()

def get_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    return None

def fetch_video_details(video_id):
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

def fetch_comments(video_id):
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
# MAIN UI
# ------------------------------
url = st.text_input("Enter YouTube video URL:")

if url:
    video_id = get_video_id(url)
    if not video_id:
        st.error("⚠️ Invalid YouTube URL")
    else:
        st.info("Fetching video details and comments...")
        details = fetch_video_details(video_id)
        df = fetch_comments(video_id)

        if details and not df.empty:
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
            # COMMENT ANALYSIS
            # ------------------------------
            df["Cleaned_Text"] = df["Text"].apply(clean_text)
            df["Polarity"] = df["Cleaned_Text"].apply(lambda x: TextBlob(x).sentiment.polarity)
            df["Sentiment"] = df["Polarity"].apply(lambda x: "Positive" if x > 0.2 else ("Negative" if x < -0.2 else "Neutral"))
            df["PublishedAt"] = pd.to_datetime(df["PublishedAt"])

            # Download option
            csv = df.to_csv(index=False)
            st.download_button("⬇️ Download Comments CSV", csv, "youtube_comments.csv", "text/csv")

            # ------------------------------
            # COMMENT ACTIVITY OVER TIME
            # ------------------------------
            st.subheader("📈 Comment Activity Over Time")
            time_data = df.groupby(df["PublishedAt"].dt.date).size()
            fig, ax = plt.subplots()
            ax.plot(time_data.index, time_data.values, marker='o')
            ax.set_xlabel("Date")
            ax.set_ylabel("Number of Comments")
            ax.set_title("Comment Frequency Over Time")
            st.pyplot(fig)

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
            pos_comments = df[df["Polarity"] > 0.5].sort_values("Likes", ascending=False).head(5)
            neg_comments = df[df["Polarity"] < -0.5].sort_values("Likes", ascending=False).head(5)

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
            # SENTIMENT DISTRIBUTION
            # ------------------------------
            st.subheader("📊 Sentiment Distribution")
            sentiment_counts = df["Sentiment"].value_counts()
            fig2, ax2 = plt.subplots()
            ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct="%1.1f%%", startangle=90)
            ax2.set_title("Overall Sentiment Breakdown")
            st.pyplot(fig2)

        else:
            st.warning("No comments found or video details unavailable.")
