# 📊 Twitter Engagement & Sentiment Dashboard

This application integrates real-time data from Twitter to perform sentiment analysis and engagement forecasting. The application fetches tweets containing the term "iPhone", which can be any kind of product or keyword. Utilizing Twitter data and Google Trends, I explore how the product is perceived in terms of engagement, sentiment and interest over time. I combine sentiment analysis from tweets with trend analysis from Google. It also offers region-based insights and engagement metrics. I provide insights into consumer sentiment, engagement patterns and geographic interest. These insights can help professionals make informed decisions by identifying emerging products before they are on the market, spotting early trends and patterns and by evaluating a potential for a product to become a market hype. 

## Table of Contents

- [Features](#features)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Database Setup](#database-setup)  
  - [1. Elasticsearch](#1-elasticsearch)  
  - [2. SQLite](#2-sqlite)  
  - [3. Mysql](#3mysql)  
- [Running](#running)  
- [Running the Dashboard](#running-the-dashboard)  
- [Project Structure](#project-structure)  


---

## Features

- Fetches recent tweets via Twitter API v2  
- Computes engagement metrics (likes, retweets, clicks, adjusted engagement, sentiment-weighted, etc.)  
- Stores data in:
  1. **Elasticsearch** 
  2. **SQLite** 
  3. **PostgreSQL** 
- Interactive Streamlit dashboard with:
  - Time-series forecasts (Prophet)
  - Engagement over time by metric & location
  - Hashtag engagement tables
  - Save & view “saved” metrics

---

## Prerequisites


- Here are the external Python libraries this project uses:
- pandas
- matplotlib
- textblob
- tweepy
- elasticsearch & elasticsearch-dsl
- prophet
- streamlit
---

## Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo

   # Install the dependencies from the requirements file
pip install -r requirements.txt


## Database Setup

This application uses three databases: **Elasticsearch**, **SQLite**, and **(Optional) MySQL/PostgreSQL** for storing and querying data. Follow the steps below to set up each of them.



### 1. Elasticsearch
Used for fast querying and full-text search of tweet data.

Download & Extract Elasticsearch (v8.x):
      ```bash
      # Download Elasticsearch
      wget [https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.6.2-linux-x86_64.tar.gz](https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.6.2-linux-x86_64.tar.gz)
      # Change into the directory
      cd elasticsearch-8.6.2/

Start elasticsearch

### Start Elasticsearch in the background
        ```bash
        bin/elasticsearch &

Wait a few seconds, then verify it's running:

        ```bash
        curl -X GET "localhost:9200"
        
✅ If successful, you'll see Elasticsearch's JSON response and can also open http://localhost:9200 in your browser. In the code I use an orm.


### SQLite
No installation is needed. A file named engagement_data.db will be automatically created in the root folder when you first run the app.

### MySQL Setup 
Download and install xampp. Start apache and a mysql server.
Create a Database and the tables with sql. The create statements are inside DB.sql.

### Run the python code to get twitter and google data

Create a Twitter Developer account and copy the Bearer Token into the python file. You can get 100 posts from the twitter feed with it.

This Python script automates the process of collecting, analyzing, storing and visualizing Twitter data for a specific product (in this case, "iPhone"). Here's what it does:

Connects to the Twitter API using the Tweepy library and search tweets about a given product, excluding retweets. It is saving 100 tweets in the elasticsrach db.

Extracts key tweet data such as text, timestamps, likes, retweets, replies, user location and follower count. It also performs sentiment analysis on each tweet using TextBlob.

Calculates various engagement metrics, including adjusted engagement and influencer impact (based on follower count) and sentiment-weighted engagement.

Stores the processed tweet data in an Elasticsearch index via with an ORM.



### How to run the dashboard

To run the Streamlit dashboard, navigate to the project's root directory in your terminal and execute the following command:

        ```bash
        streamlit run dashboard.py

This will open your browser and open a webpage with the dashboard.
![image](https://github.com/user-attachments/assets/1777210c-79a5-4032-a229-8c9669b172cf)
This is a image of the dashboard.
