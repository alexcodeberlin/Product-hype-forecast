from elasticsearch_dsl import Document, Date, Text, Keyword, Integer, Float, connections

# Establish connection
connections.create_connection(hosts=["http://localhost:9200"])

INDEX_NAME = "twitter_datav7"

class TweetDocument(Document):
    tweet_id = Keyword()
    timestamp = Date()
    text = Text()
    sentiment_score = Float()
    likes = Integer()
    retweets = Integer()
    replies = Integer()
    clicks = Integer()
    user_location = Text(fields={"keyword": Keyword()})
    followers = Integer()
    regular_engagement = Integer()
    google_engagement = Float()
    high_follower_engagement = Float()
    adjusted_engagement = Float()
    engagement_including_sentiment = Float()
    engagement_final = Float()

    class Index:
        name = INDEX_NAME

def create_index():
    if not TweetDocument._index.exists():
        TweetDocument.init()
        print(f"✅ DSL index '{INDEX_NAME}' created.")
    else:
        print(f"ℹ️ DSL index '{INDEX_NAME}' already exists.")
