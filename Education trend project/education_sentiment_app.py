import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Education News Sentiment", layout="wide")
st.title("🧠 Education News Trend & Sentiment Study")
st.write("Analyze trends and emotions in recent education-related news articles.")

try:
    df = pd.read_csv("education_news_analysis.csv")
except FileNotFoundError:
    st.error("⚠️ 'education_news_analysis.csv' not found. Please run the analysis script first.")
    st.stop()

df['date'] = pd.to_datetime(df['publishedAt']).dt.date

total_articles = len(df)
avg_sentiment = round(df['sentiment'].mean(), 3)
positive_count = (df['sentiment_label'] == "Positive").sum()
neutral_count = (df['sentiment_label'] == "Neutral").sum()
negative_count = (df['sentiment_label'] == "Negative").sum()

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("📰 Total Articles", total_articles)
col2.metric("😊 Positive", positive_count)
col3.metric("😐 Neutral", neutral_count)
col4.metric("😞 Negative", negative_count)
col5.metric("📊 Avg Sentiment", avg_sentiment)

st.subheader("📈 Sentiment Trend Over Time")
daily_sentiment = df.groupby('date')['sentiment'].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=daily_sentiment, x='date', y='sentiment', marker='o', color='blue', ax=ax)
ax.set_title("Education News Sentiment Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Average Sentiment (VADER Score)")
plt.xticks(rotation=45)
st.pyplot(fig)

st.subheader("📊 Sentiment Distribution")

sentiment_counts = df['sentiment_label'].value_counts()
fig2, ax2 = plt.subplots(figsize=(5, 3))
ax2.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
ax2.axis('equal')
st.pyplot(fig2)

st.subheader("🌟 Highlighted Headlines")

most_positive = df.loc[df['sentiment'].idxmax()]
most_negative = df.loc[df['sentiment'].idxmin()]

st.markdown("### 🌞 Most Positive Headline")
st.markdown(f"**🏷️ Title:** {most_positive['title']}")
st.markdown(f"**📰 Source:** {most_positive['source']}")
st.markdown(f"**🔗 [Read Article]({most_positive['url']})**")

st.markdown("### ⚡ Most Negative Headline")
st.markdown(f"**🏷️ Title:** {most_negative['title']}")
st.markdown(f"**📰 Source:** {most_negative['source']}")
st.markdown(f"**🔗 [Read Article]({most_negative['url']})**")

st.subheader("🔍 Full Article Sentiment Data")
st.dataframe(df[['date', 'source', 'title', 'sentiment_label', 'sentiment', 'url']])

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "📘 Education News Sentiment Dashboard | Built with ❤️ using Python & Streamlit"
    "</div>",
    unsafe_allow_html=True
)