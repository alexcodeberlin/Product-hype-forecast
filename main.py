# Import necessary libraries
import time 
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob  # Used for sentiment analysis
import tweepy  # Twitter API client
from tweepy.errors import TooManyRequests  # Handle Twitter rate limits

# Import custom modules
from models import create_index  # Function to create Elasticsearch index
from save_dsl import save_to_elasticsearch_dsl  # Function to save data to Elasticsearch

# --------------------
# Configuration
# --------------------

# Twitter API Bearer Token (authentication)
BEARER_TOKEN = "..."  # Replace with your real token
PRODUCT = "iPhone"  # Keyword to search tweets for

# --------------------
# Initialize Tweepy Client
# --------------------

# Create Twitter API client using bearer token
client = tweepy.Client(bearer_token=BEARER_TOKEN)

# --------------------
# Functions
# --------------------

# Function to fetch tweets related to a product
def fetch_twitter_data(product, max_tweets=10):
    print(f"\n🔍 Fetching up to {max_tweets} tweets for '{product}'...\n")
    try:
        # Search recent English tweets, excluding retweets
        resp = client.search_recent_tweets(
            query=f"{product} lang:en -is:retweet",
            max_results=max_tweets,
            tweet_fields=["created_at","public_metrics","author_id"],
            user_fields=["location","public_metrics"],
            expansions=["author_id"]
        )
    except TooManyRequests as e:
        # If rate limit is hit, wait until it resets and retry
        reset = int(e.response.headers.get("x-rate-limit-reset", time.time()))
        wait  = reset - int(time.time()) + 1
        print(f"⏱ Rate limit, waiting {wait}s…")
        time.sleep(wait)
        return fetch_twitter_data(product, max_tweets)

    # If no data found, return empty DataFrame
    if not resp.data:
        return pd.DataFrame()

    # Extract user info: location and followers
    users = {
        u.id: {
            "location": u.location or "Unknown",
            "followers": u.public_metrics["followers_count"]
        }
        for u in resp.includes.get("users", [])
    }

    # Process each tweet into a row of data
    rows = []
    for t in resp.data:
        u = users.get(t.author_id, {"location":"Unknown","followers":0})
        rows.append({
            "tweet_id":         str(t.id),
            "timestamp":        t.created_at,
            "text":             t.text,
            "sentiment_score":  TextBlob(t.text).sentiment.polarity,  # Sentiment polarity score
            "likes":            t.public_metrics.get("like_count", 0),
            "retweets":         t.public_metrics.get("retweet_count", 0),
            "replies":          t.public_metrics.get("reply_count", 0),
            "clicks":           int(t.public_metrics.get("like_count", 0) * 0.1),  # Estimate clicks
            "user_location":    u["location"],
            "followers":        u["followers"]
        })

    return pd.DataFrame(rows)

# Function to calculate engagement metrics
def add_engagement_metrics(df, amp=1.5):
    if df.empty:
        return df

    # Sum of basic interaction metrics
    df["regular_engagement"] = df[["likes", "retweets", "replies", "clicks"]].sum(axis=1)
    df["google_engagement"] = 0.0  # Placeholder for possible future metric

    # Boost engagement if user has many followers
    df["high_follower_engagement"] = df["regular_engagement"] * (df["followers"] >= 10000) * amp
    
    # Adjust engagement by follower influence
    df["adjusted_engagement"] = df["regular_engagement"] + df["high_follower_engagement"]
    
    # Include sentiment as a multiplier for engagement
    df["engagement_including_sentiment"] = df["adjusted_engagement"] * (1 + df["sentiment_score"])

    # Final engagement is a sum of all components
    df["engagement_final"] = (
        df["regular_engagement"] +
        df["google_engagement"] +
        df["high_follower_engagement"] +
        df["adjusted_engagement"] +
        df["engagement_including_sentiment"]
    )

    return df

# Function to visualize engagement over time
def plot_twitter_engagement(df, title):
    if df.empty:
        print("❌ Nothing to plot.")
        return

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Create line plot for engagement
    plt.figure(figsize=(12, 6))
    plt.plot(df["timestamp"], df["regular_engagement"], label="Regular Engagement")
    plt.plot(df["timestamp"], df["google_engagement"],  label="Google Engagement")
    plt.xlabel("Time")
    plt.ylabel("Engagement")
    plt.title(f"Engagement Trends – {title}")
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --------------------
# Main Script
# --------------------

if __name__ == "__main__":
    create_index()  # Set up Elasticsearch index if not already present

    # Fetch tweets and enrich with engagement metrics
    tweets_df = fetch_twitter_data(PRODUCT, max_tweets=100)
    tweets_df = add_engagement_metrics(tweets_df)

    # Save processed data to Elasticsearch
    save_to_elasticsearch_dsl(tweets_df)

    # Display engagement trends using a plot
    plot_twitter_engagement(tweets_df, PRODUCT)
