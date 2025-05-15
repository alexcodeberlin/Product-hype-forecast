# Import required libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from elasticsearch_dsl import Document, Date, Float, Integer, Keyword, Text, connections
from collections import defaultdict, Counter
import re
import sqlite3

import mysql.connector
import hashlib
from datetime import datetime

# Connect to MySQL database
def connect_to_db():
    try:
        return mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="user_plot_app"
        )
    except mysql.connector.Error as err:
        st.error(f"MySQL connection failed: {err}")
        st.stop()

# --------------------
# Elasticsearch Configuration
# --------------------
ES_HOST = "http://localhost:9200"  # Elasticsearch server URL
ES_INDEX = "twitter_datav7"        # Index name for tweets in Elasticsearch

# Connect to the Elasticsearch server
connections.create_connection(hosts=[ES_HOST])

# --------------------
# Define Elasticsearch ORM Model
# --------------------
# This class defines the structure of documents (tweets) stored in Elasticsearch
class Tweet(Document):
    tweet_id = Keyword()
    timestamp = Date()
    sentiment_score = Float()
    likes = Integer()
    retweets = Integer()
    replies = Integer()
    clicks = Integer()
    text = Text()
    user_location = Text()
    followers = Integer()
    regular_engagement = Integer()
    google_engagement = Float()
    high_follower_engagement = Float()
    adjusted_engagement = Float()
    engagement_including_sentiment = Float()
    engagement_final = Float()
        
    class Index:
        name = ES_INDEX
        settings = {
            "number_of_shards": 3,
            "number_of_replicas": 2
        }

# --------------------
# SQLite Database Setup
# --------------------
DB_FILE = "engagement_data.db"  # SQLite database file

# Create the database table if it doesn't exist
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS engagement_metrics5 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            xAxis TEXT,
            yAxis REAL
        )
    ''')
    conn.commit()
    conn.close()

# Save data (x, y values) into the database
def save_to_db(x_values, y_values):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    for x, y in zip(x_values, y_values):
        cursor.execute(
            "INSERT INTO engagement_metrics5 (xAxis, yAxis) VALUES (?, ?)",
            (x, y)
        )
    conn.commit()
    conn.close()

# Fetch all saved engagement data from the database
def get_saved_data():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM engagement_metrics5")
    data = cursor.fetchall()
    conn.close()
    return data

# Initialize the database on app start
init_db()

# --------------------
# Load Data Functions
# --------------------

# Load Twitter sentiment data (timestamp and sentiment score)
@st.cache_data
def load_twitter_data():
    query = Tweet.search()[:10000]
    response = query.execute()

    records = []
    for hit in response:
        if hasattr(hit, 'timestamp') and hasattr(hit, 'sentiment_score'):
            records.append({"ds": hit.timestamp, "y": hit.sentiment_score})
    df = pd.DataFrame(records)
    if not df.empty:
        df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    return df

# Load engagement data for a specific metric (optionally filtered by location)
@st.cache_data
def load_engagement_data(metric, location=None):
    query = Tweet.search()[:10000]
    response = query.execute()

    records = []
    for hit in response:
        if hasattr(hit, 'timestamp') and hasattr(hit, metric):
            loc = getattr(hit, "user_location", "").strip().lower()
            if location is None or loc == location:
                records.append({"ds": hit.timestamp, "y": getattr(hit, metric)})
    df = pd.DataFrame(records)
    if not df.empty:
        df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    return df

# Load final engagement metric data
@st.cache_data
def load_engagement_final():
    query = Tweet.search()[:10000]
    response = query.execute()

    records = []
    for hit in response:
        if hasattr(hit, 'timestamp') and hasattr(hit, 'engagement_final'):
            records.append({"ds": hit.timestamp, "y": hit.engagement_final})
    df = pd.DataFrame(records)
    if not df.empty:
        df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    return df

# Get a list of unique user locations with counts
@st.cache_data
def get_unique_user_locations():
    query = Tweet.search()[:10000]
    response = query.execute()
    locations = [getattr(hit, "user_location", "").strip().lower() for hit in response if hasattr(hit, "user_location")]
    location_counts = Counter(locations)
    cleaned_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)
    return [f"{loc.title()} ({count})" for loc, count in cleaned_locations]

# --------------------
# Hashtag Engagement Table
# --------------------

# Extract hashtags from tweets and calculate their average engagement
@st.cache_data
def get_hashtag_engagement_data():
    query = Tweet.search()[:10000]
    response = query.execute()

    hashtag_engagement = defaultdict(list)

    for hit in response:
        text = getattr(hit, "text", "")
        engagement = getattr(hit, "engagement_including_sentiment", None)
        if not text or engagement is None:
            continue

        hashtags = re.findall(r"#\w+", text)
        for tag in hashtags:
            hashtag_engagement[tag.lower()].append(engagement)

    # Create DataFrame of average engagement per hashtag
    table_data = [{"Hashtag": tag, "Avg Engagement": sum(vals) / len(vals)} for tag, vals in hashtag_engagement.items() if vals]
    df = pd.DataFrame(table_data).sort_values("Avg Engagement", ascending=False)
    return df

# --------------------
# Plotting Functions
# --------------------

# Plot historical data
def plot_past_data(df, title, ylabel):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["ds"], df["y"], marker="o", linestyle="-")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    st.pyplot(fig)

# Plot forecast with Prophet results
def plot_forecast_data(df, forecast, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["ds"], df["y"], marker="o", label="Past Data")
    ax.plot(forecast["ds"], forecast["yhat"], linestyle="dashed", label="Forecast")
    ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], alpha=0.3)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Metric")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)

# --------------------
# Streamlit App Interface
# --------------------

# Hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Save user to database
def register_user(username, email, password):
    try:
        conn = connect_to_db()
        cursor = conn.cursor()

        password_hashed = hash_password(password)
        query = """
            INSERT INTO users (username, email, password_hash, created_at)
            VALUES (%s, %s, %s, %s)
        """
        values = (username, email, password_hashed, datetime.now())

        cursor.execute(query, values)
        conn.commit()

        return True, "User registered successfully!"
    except mysql.connector.Error as err:
        return False, f"Error: {err}"
    finally:
        cursor.close()
        conn.close()
st.title("📊 Future Trend & Sentiment Prediction")

# Sidebar options to select the type of data
dataset_choice = st.sidebar.radio(
    "Select Dataset:",
    ["Google Trends", "Twitter Sentiment", "Engagement Overview", "Favourite Overview", "Register and Login", "Shared plots"]
)

# Show slider only for datasets that require forecasting
if dataset_choice in ["Google Trends", "Twitter Sentiment", "Engagement Overview"]:
    forecast_seconds = st.sidebar.slider("Select number of seconds to predict:", 30, 3600, 1800, 30)

# Placeholder if Google Trends data were to be added
if dataset_choice == "Google Trends":
    pass

# Process and forecast Twitter sentiment
elif dataset_choice == "Twitter Sentiment":
    df = load_twitter_data()
    st.subheader("💬 Predicting Twitter Sentiment for iPhone Tweets")

    if df.empty:
        st.warning("No data available for Twitter Sentiment.")
    else:
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=forecast_seconds, freq="30S")
        forecast = model.predict(future)

        plot_forecast_data(df, forecast, f"Predicted Twitter Sentiment for the Next {forecast_seconds} Seconds")
        st.write(f"### Forecasted Data (Next {forecast_seconds} Seconds)")
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_seconds))

# Show and forecast various engagement metrics
elif dataset_choice == "Engagement Overview":
    st.subheader("📣 Past Engagement Metrics for iPhone Tweets")

    user_locations = get_unique_user_locations()
    selected_display = st.selectbox("Filter by User Location", ["All"] + user_locations)
    selected_location = None if selected_display == "All" else selected_display.split(" (")[0].strip().lower()

    df_final = load_engagement_final()
    if df_final.empty:
        st.warning("No data available for Engagement Final.")
    else:
        plot_past_data(df_final, "Engagement Final Over Time", "Engagement Final")

        # Button to save current data into the SQLite database
        if st.button("🔍 Save Engagement Final Data to Database"):
            x_values = df_final["ds"].astype(str)
            y_values = df_final["y"]
            save_to_db(x_values, y_values)
            st.success("Data successfully saved to the database!")

        # Forecast engagement_final
        model = Prophet()
        model.fit(df_final)
        future = model.make_future_dataframe(periods=forecast_seconds, freq="30S")
        forecast = model.predict(future)

        plot_forecast_data(df_final, forecast, f"Forecasted Engagement Final for the Next {forecast_seconds} Seconds")
        st.write(f"### Forecasted Data (Next {forecast_seconds} Seconds) for Engagement Final")
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_seconds))

    # Loop through and display other engagement metrics
    metrics = [
        "regular_engagement",
        "google_engagement",
        "adjusted_engagement",
        "engagement_including_sentiment",
        "high_follower_engagement"
    ]

    for metric in metrics:
        df_metric = load_engagement_data(metric, selected_location)
        if df_metric.empty:
            st.warning(f"No data available for {metric.replace('_', ' ').title()} at the selected location.")
        else:
            plot_past_data(df_metric, f"{metric.replace('_', ' ').title()} Over Time", metric.replace('_', ' ').title())

            model = Prophet()
            model.fit(df_metric)
            future = model.make_future_dataframe(periods=forecast_seconds, freq="30S")
            forecast = model.predict(future)

            plot_forecast_data(df_metric, forecast, f"Forecasted {metric.replace('_', ' ').title()} for the Next {forecast_seconds} Seconds")
            st.write(f"### Forecasted Data (Next {forecast_seconds} Seconds) for {metric.replace('_', ' ').title()}")
            st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_seconds))

    # Display top hashtags by engagement
    st.subheader("🏷️ Hashtag Engagement Table")
    df_hashtags = get_hashtag_engagement_data()
    if df_hashtags.empty:
        st.write("No hashtag data found.")
    else:
        st.dataframe(df_hashtags.reset_index(drop=True))

# Show saved engagement data from SQLite
elif dataset_choice == "Favourite Overview":
    st.subheader("🔖 Favourites Overview")

    saved_data = get_saved_data()

    if saved_data:
        df_saved_data = pd.DataFrame(saved_data, columns=["ID", "xAxis", "yAxis"])
        df_saved_data['xAxis'] = pd.to_datetime(df_saved_data['xAxis'], errors='coerce')

        if df_saved_data['xAxis'].isnull().any():
            st.warning("Some xAxis values could not be converted to datetime.")

        st.write("### Saved Engagement Data")
        st.dataframe(df_saved_data)

        # Plot saved data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df_saved_data['xAxis'], df_saved_data['yAxis'], marker="o", linestyle="-")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Engagement Metric")
        ax.set_title("Saved Engagement Metrics Over Time")
        st.pyplot(fig)

    else:
        st.write("No data found in the database.")

elif dataset_choice == "Register and Login":

    st.subheader("🔐 Register and Login")
    tab1, tab2 = st.tabs(["Register", "Login"])

    with tab1:
        st.subheader("Create a new account")
        
        username = st.text_input("Username", key="register_username")
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm")

        if st.button("Register"):
            if password != confirm_password:
                st.error("Passwords do not match.")
            elif not username or not email or not password:
                st.error("Please fill in all fields.")
            else:
                success, message = register_user(username, email, password)
                if success:
                    st.success(message)
                else:
                    st.error(message)

    with tab2:
        st.subheader("Login (not implemented yet)")
        st.info("Login functionality will be added later.")
