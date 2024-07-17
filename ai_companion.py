import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import os
from dotenv import load_dotenv
import sqlite3


load_dotenv()


@st.cache_resource
def load_models():
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    return sentence_model, ner_model

sentence_model, ner_model = load_models()


OPENTRIPMAP_API_KEY = os.getenv("OPENTRIPMAP_API_KEY")

def get_db_connection():
    conn = sqlite3.connect('travel_destinations.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS destinations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  name TEXT,
                  city TEXT,
                  country TEXT,
                  description TEXT,
                  rate REAL,
                  embedding TEXT)''')
    conn.commit()
    conn.close()

def fetch_destinations_from_api(limit=100):
    url = f"http://api.opentripmap.com/0.1/en/places/radius?radius=1000000&lon=0&lat=0&rate=3&format=json&limit={limit}&apikey={OPENTRIPMAP_API_KEY}&sort=rate"
    response = requests.get(url)
    if response.status_code == 200:
        places = response.json()
        destinations = []
        for place in places:
            detail_url = f"http://api.opentripmap.com/0.1/en/places/xid/{place['xid']}?apikey={OPENTRIPMAP_API_KEY}"
            detail_response = requests.get(detail_url)
            if detail_response.status_code == 200:
                detail = detail_response.json()
                destinations.append({
                    "name": detail.get('name', 'Unknown'),
                    "city": detail.get('address', {}).get('city', 'Unknown'),
                    "country": detail.get('address', {}).get('country', 'Unknown'),
                    "description": detail.get('wikipedia_extracts', {}).get('text', 'No description available.'),
                    "rate": place.get('rate', 0)
                })
        return destinations
    else:
        st.error("Failed to fetch destinations from OpenTripMap API")
        return []

@st.cache_resource
def initialize_database():
    init_db()
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM destinations")
    if c.fetchone()[0] == 0:
        destinations = fetch_destinations_from_api()
        for dest in destinations:
            embedding = sentence_model.encode(dest['description']).tolist()
            c.execute('''INSERT INTO destinations (name, city, country, description, rate, embedding)
                         VALUES (?, ?, ?, ?, ?, ?)''',
                      (dest['name'], dest['city'], dest['country'], dest['description'], dest['rate'], json.dumps(embedding)))
        conn.commit()
    conn.close()

initialize_database()

def get_destinations_from_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM destinations ORDER BY rate DESC")
    rows = c.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=['id', 'name', 'city', 'country', 'description', 'rate', 'embedding'])
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

def add_destination_to_db(name, city, country, description, rate):
    conn = get_db_connection()
    c = conn.cursor()
    embedding = sentence_model.encode(description).tolist()
    c.execute('''INSERT INTO destinations (name, city, country, description, rate, embedding)
                 VALUES (?, ?, ?, ?, ?, ?)''',
              (name, city, country, description, rate, json.dumps(embedding)))
    conn.commit()
    conn.close()

destinations_df = get_destinations_from_db()



def get_weather(city):
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['main']['temp'], data['weather'][0]['description']
    except requests.RequestException as e:
        st.error(f"Error fetching weather data: {e}")
        return None, None
def get_currency_code(country_name):
    with open('currency_code.json', 'r', encoding='utf-8') as file:
        code = json.load(file)
    for entry in code:
        if entry['country'].lower() == country_name.lower():
            return entry['currency_code']
    return None
def get_exchange_rate(base_currency, target_currency):
    target_currency = get_currency_code(target_currency)
    api_key = os.getenv("EXCHANGERATE_API_KEY")
    url = f"https://v6.exchangerate-api.com/v6/{api_key}/latest/{base_currency}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        return data['conversion_rates'].get(target_currency, None)
    except requests.RequestException as e:
        st.error(f"Error fetching exchange rate data: {e}")
        return None

# Utility functions
def find_similar_destinations(query, top_k=3):
    query_embedding = sentence_model.encode([query])
    similarities = cosine_similarity(query_embedding, np.vstack(destinations_df['embedding']))
    top_indices = similarities[0].argsort()[-top_k:][::-1]
    return destinations_df.iloc[top_indices]

def extract_locations(text):
    entities = ner_model(text)
    locations = [entity['word'] for entity in entities if entity['entity'] == 'LOC']
    return list(set(locations))


st.title("AI-Powered Travel Companion")

user_input = st.text_input("What kind of destination are you interested in?")

if user_input:
    st.write("Processing your request...")
    

    similar_destinations = find_similar_destinations(user_input)
    

    locations = extract_locations(user_input)
    
    st.subheader("Recommended Destinations")
    for _, dest in similar_destinations.iterrows():
        with st.expander(f"{dest['name']}, {dest['city']}, {dest['country']}"):
            st.write(dest['description'])
            

            temp, weather_desc = get_weather(dest['city'])
            if temp and weather_desc:
                st.write(f"Current weather: {temp:.1f}°C, {weather_desc}")
            else:
                st.write("Weather information unavailable")
            

            rate = get_exchange_rate("USD", dest['country'])
            if rate:
                st.write(f"Exchange rate: 1 USD = {rate:.2f} {get_currency_code(dest['country'])}")
            else:
                st.write("Exchange rate information unavailable")
    
    if locations:
        st.subheader("Specific Locations Mentioned")
        st.write(f"I noticed you mentioned {', '.join(locations)}. Here's some information about these locations:")
        for location in locations:
            matching_dests = destinations_df[destinations_df['name'].str.contains(location) | 
                                             destinations_df['city'].str.contains(location) | 
                                             destinations_df['country'].str.contains(location)]
            if not matching_dests.empty:
                for _, dest in matching_dests.iterrows():
                    st.write(f"- {dest['name']}, {dest['city']}, {dest['country']}")
                    st.write(f"  {dest['description']}")
            else:
                st.write(f"- {location}: No specific information available in our database.")

st.sidebar.header("Add New Destination")
new_name = st.sidebar.text_input("Name")
new_city = st.sidebar.text_input("City")
new_country = st.sidebar.text_input("Country")
new_description = st.sidebar.text_area("Description")
new_rate = st.sidebar.slider("Rate", 1.0, 7.0, 5.0, 0.1)

if st.sidebar.button("Add Destination"):
    add_destination_to_db(new_name, new_city, new_country, new_description, new_rate)
    st.sidebar.success("Destination added successfully!")

    destinations_df = get_destinations_from_db()

st.sidebar.title("More Options")

if st.sidebar.checkbox("Show All Destinations"):
    st.sidebar.table(destinations_df[['name', 'city', 'country', 'rate']])

st.sidebar.subheader("Currency Converter")
amount = st.sidebar.number_input("Amount", min_value=0.01, value=1.0, step=0.01)
from_currency = st.sidebar.selectbox("From Currency", ["USD", "EUR", "GBP", "JPY"])
to_currency = st.sidebar.selectbox("To Currency", ["EUR", "GBP", "JPY", "USD"])

if st.sidebar.button("Convert"):
    rate = get_exchange_rate(from_currency, to_currency)
    if rate:
        converted_amount = amount * rate
        st.sidebar.write(f"{amount} {from_currency} = {converted_amount:.2f} {to_currency}")
    else:
        st.sidebar.write("Conversion failed. Please try again later.")

st.sidebar.subheader("Weather Lookup")
city_for_weather = st.sidebar.text_input("Enter city name")
if st.sidebar.button("Get Weather"):
    temp, weather_desc = get_weather(city_for_weather)
    if temp and weather_desc:
        st.sidebar.write(f"Temperature in {city_for_weather}: {temp:.1f}°C")
        st.sidebar.write(f"Weather condition: {weather_desc}")
    else:
        st.sidebar.write("Weather information not available. Please check the city name and try again.")

st.markdown("---")