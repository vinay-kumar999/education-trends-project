import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from newsapi import NewsApiClient
import re

nltk.download('vader_lexicon')


api_key = "8d4e3c0f531b49188b5a92e71f924ec2"  
newsapi = NewsApiClient(api_key=api_key)


print("Fetching education news articles...")
all_articles = newsapi.get_everything(
    q='education OR school OR teacher OR university OR learning',
    language='en',
    from_param='2025-11-01',
    to='2025-11-11',
    sort_by='relevancy',
    page_size=100
)

articles = pd.DataFrame(all_articles['articles'])
articles = articles[['source', 'author', 'title', 'description', 'content', 'publishedAt', 'url']]
articles['source'] = articles['source'].apply(lambda x: x['name'] if isinstance(x, dict) else x)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

articles['clean_text'] = articles['title'].astype(str) + " " + articles['description'].astype(str)
articles['clean_text'] = articles['clean_text'].apply(clean_text)

sid = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if not text.strip():
        return 0
    return sid.polarity_scores(text)['compound']

articles['sentiment'] = articles['clean_text'].apply(get_sentiment)

def label_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

articles['sentiment_label'] = articles['sentiment'].apply(label_sentiment)

articles['publishedAt'] = pd.to_datetime(articles['publishedAt'])
articles['date'] = articles['publishedAt'].dt.date
daily_sentiment = articles.groupby('date')['sentiment'].mean().reset_index()

sns.set(style='whitegrid')
plt.figure(figsize=(10,5))
sns.lineplot(data=daily_sentiment, x='date', y='sentiment', marker='o', color='blue')
plt.title('📈 Education News Sentiment Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment (VADER Score)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("sentiment_trend.png")
plt.close()

avg_sentiment = round(articles['sentiment'].mean(), 3)
positive_count = (articles['sentiment_label'] == "Positive").sum()
neutral_count = (articles['sentiment_label'] == "Neutral").sum()
negative_count = (articles['sentiment_label'] == "Negative").sum()
total_articles = len(articles)

most_pos = articles.loc[articles['sentiment'].idxmax()]
most_neg = articles.loc[articles['sentiment'].idxmin()]

summary = f"""
============================================================
🧠 Education News Sentiment Analysis Summary
============================================================
📅 Analysis Period : 2025-11-01 to 2025-11-11
📰 Total Articles  : {total_articles}
😊 Positive News   : {positive_count}
😐 Neutral News    : {neutral_count}
😞 Negative News   : {negative_count}
📊 Average Sentiment Score : {avg_sentiment}

🌟 Most Positive Headline:
   🏷️ Title  : {most_pos['title']}
   📰 Source : {most_pos['source']}
   🔗 URL    : {most_pos['url']}

⚡ Most Negative Headline:
   🏷️ Title  : {most_neg['title']}
   📰 Source : {most_neg['source']}
   🔗 URL    : {most_neg['url']}

============================================================
📈 Visualization saved as: sentiment_trend.png
💾 Detailed data saved as: education_news_analysis.csv
📄 Summary report saved as: output.txt
============================================================
"""

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(summary)

print(summary)

articles.to_csv("education_news_analysis.csv", index=False)
print("\n✅ Results saved to education_news_analysis.csv")