import random
from pytrends.request import TrendReq
import pandas as pd
import time
import matplotlib.pyplot as plt

def fetch_google_trends(product, country):
    """Fetch Google Trends data for a product in a given country."""
    pytrends = TrendReq(hl="en-US", tz=360)

    kw_list = [product]
    print(f"Fetching Google Trends data for '{product}' in {country}...")

    try:
        # Use a longer timeframe to get meaningful data
        pytrends.build_payload(kw_list, timeframe="today 3-m", geo=country)
        
        # Get interest over time
        interest_over_time = pytrends.interest_over_time()
        
        # Ensure data is available
        if interest_over_time.empty:
            print(f"⚠️ No data found for {product} in {country}")
            return None, None, None

        # Safely get trending queries, avoiding index errors
        trending_queries = None
        try:
            trending_queries = pytrends.related_queries()[product]["top"]
        except (KeyError, IndexError):
            print(f"⚠️ No trending queries found for {product}")

        if trending_queries is None or trending_queries.empty:
            print(f"⚠️ No trending queries data for {product}")
            trending_queries = pd.DataFrame()  # Ensure it is an empty DataFrame if no data

        # Get interest by region
        interest_by_region = pytrends.interest_by_region()

        # Ensure region data is available
        if interest_by_region.empty:
            print(f"⚠️ No region-based data found for {product}")
            interest_by_region = pd.DataFrame()  # Empty DataFrame if no region data

        return interest_over_time, trending_queries, interest_by_region

    except Exception as e:
        print(f"⚠️ Error: {e}")
        return None, None, None

def calculate_percentage_increase(data):
    """Calculate the percentage increase in search interest every 2 days."""
    data["% Increase"] = data["iPhone"].pct_change(periods=2) * 100  # Change over 2 days
    return data

def save_to_csv(interest_data, trending_data, region_data, product):
    """Save the data to CSV files."""
    if interest_data is not None:
        interest_data.to_csv(f"{product}_trends.csv", index=True)
        print(f"✅ Saved search interest data: {product}_trends.csv")
    
    if trending_data is not None:
        trending_data.to_csv(f"{product}_trending_queries.csv", index=False)
        print(f"✅ Saved trending queries data: {product}_trending_queries.csv")
    
    if region_data is not None:
        region_data.to_csv(f"{product}_interest_by_region.csv", index=False)
        print(f"✅ Saved region-based data: {product}_interest_by_region.csv")

def plot_percentage_increase(data, product):
    """Plot the percentage increase of search interest over time."""
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data["% Increase"], marker='o', linestyle='-', color='b', label="Percentage Increase")
    plt.title(f"Percentage Increase in Google Trends for '{product}'")
    plt.xlabel("Date")
    plt.ylabel("Percentage Increase")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_region_interest(region_data, product):
    """Plot the interest by region."""
    if region_data is not None and not region_data.empty:
        # Print the columns of region_data to inspect it
        print("Region Data Columns:", region_data.columns)
        
        # Check if the 'geoName' column exists
        if 'geoName' in region_data.columns:
            region_data_sorted = region_data.sort_values(by="iPhone", ascending=False)
            plt.bar(region_data_sorted["geoName"], region_data_sorted["iPhone"], color="green")
        else:
            # If 'geoName' column is not found, inspect the first few rows of region data
            print("⚠️ 'geoName' column not found. Inspecting first few rows of region data:")
            print(region_data.head())
            
            # Handle the case where there is only one column in the data (no 'geoName' column)
            if len(region_data.columns) == 1:
                region_data_sorted = region_data.sort_values(by=region_data.columns[0], ascending=False)
                plt.bar(region_data_sorted.index, region_data_sorted[region_data.columns[0]], color="green")
            else:
                print("⚠️ Unexpected region data structure. Unable to plot.")
                return

        plt.title(f"Interest by Region for '{product}'")
        plt.xlabel("Region")
        plt.ylabel("Interest Level")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ No region data available to plot.")

def fetch_trending_products(country):
    """Fetch the top 10 trending products in the past 7 days."""
    pytrends = TrendReq(hl="en-US", tz=360)

    print(f"Fetching top 10 trending products in {country}...")

    try:
        pytrends.build_payload(kw_list=[], timeframe="now 7-d", geo=country)
        trending_searches = pytrends.trending_searches(pn=country)  # Fetch trending searches for the last 7 days
        trending_searches = trending_searches.head(10)  # Get top 10 trending products

        if trending_searches.empty:
            print("⚠️ No trending products found.")
            return pd.DataFrame()  # Return empty DataFrame if no trending products found

        return trending_searches

    except Exception as e:
        print(f"⚠️ Error: {e}")
        return pd.DataFrame()

# Run the script
product = "iPhone"  # You can change this
country = "DE"  # Example: "US", "GB", "IN"

# Introduce delay to avoid rate limiting
time.sleep(random.uniform(5, 15))  # Wait between 5 to 15 seconds

# Fetch data for the specific product
trends_data, trending_queries, region_data = fetch_google_trends(product, country)

# Fetch the top 10 trending products in the last 7 days
top_trending_products = fetch_trending_products(country)

# Save the trending products
top_trending_products.to_csv(f"{country}_top_trending_products.csv", index=False)
print(f"✅ Saved top trending products: {country}_top_trending_products.csv")

# If data exists, process and save it
if trends_data is not None:
    # Calculate the percentage increase
    trends_data = calculate_percentage_increase(trends_data)
    
    # Save data to CSV
    save_to_csv(trends_data, trending_queries, region_data, product)
    
    # Print first few rows
    print(trends_data.head())  
    
    # Plot the percentage increase over time
    plot_percentage_increase(trends_data, product)
    
    # Plot the region interest data
    plot_region_interest(region_data, product)
