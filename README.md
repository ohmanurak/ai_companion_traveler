# AI Travel Companion MVP

## Overview and Use Case
The AI Travel Companion is a Minimum Viable Product (MVP) designed to assist travelers in planning their trips and exploring destinations. It uses artificial intelligence to recommend destinations based on user input, provide weather information, and offer currency exchange rates.

Key features:
- Destination recommendation based on user preferences
- Real-time weather information for destinations
- Currency conversion
- Named Entity Recognition for location extraction from user input

Use Case: Travelers can use this application to discover new destinations that match their interests, get up-to-date weather information, and quickly convert currencies for their trip planning.

## AI Techniques and Tools Used
1. Sentence Transformers: For text embedding and similarity search
2. Hugging Face Transformers: For Named Entity Recognition
3. Streamlit: For creating the user interface
4. OpenTripMap API: For fetching destination data
5. OpenWeatherMap API: For real-time weather information
6. ExchangeRate-API: For currency conversion rates

## Embedding Explanation
I use the Sentence Transformers model 'all-MiniLM-L6-v2' to create embeddings for destination descriptions. These embeddings are vector representations of the text that capture semantic meaning. I use these embeddings to find similar destinations based on user input by calculating cosine similarity between the input query embedding and the destination embeddings.

## Hugging Face Components
I use the Hugging Face Transformers library for Named Entity Recognition (NER). Specifically, we use the "dbmdz/bert-large-cased-finetuned-conll03-english" model to extract location names from user input. This helps in identifying specific locations that the user might be interested in.

## API Usage Details
1. OpenTripMap API:
   - Used to fetch destination data
   - Endpoint: `http://api.opentripmap.com/0.1/en/places/`
   - Requires an API key

2. OpenWeatherMap API:
   - Used to fetch current weather data for destinations
   - Endpoint: `http://api.openweathermap.org/data/2.5/weather`
   - Requires an API key

3. ExchangeRate-API:
   - Used for currency conversion
   - Endpoint: `https://v6.exchangerate-api.com/v6/`
   - Requires an API key

## Running and Testing the MVP
1. Clone the repository
2. Install required packages:
3. Create a `.env` file in the project root with the following content:
   ```
   OPENTRIPMAP_API_KEY=your_opentripmap_api_key
   OPENWEATHERMAP_API_KEY=your_openweathermap_api_key
   EXCHANGERATE_API_KEY=your_exchangerate_api_key
   ```
5. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
7. Open the provided URL in your web browser
8. Enter destination preferences in the text input
9. Explore recommended destinations, weather information, and use the currency converter

Note: Ensure you have valid API keys for OpenTripMap, OpenWeatherMap, and ExchangeRate-API.

## Brief summary of the development process:
The development of the AI Travel Companion MVP involved several key steps and decisions:

1. Choice of AI techniques: I decided to use Sentence Transformers for text embedding due to its effectiveness in capturing semantic meaning, which is crucial for destination recommendations. Hugging Face Transformers was chosen for NER due to its ease of use and pre-trained models.
2. API Integration: I integrated multiple APIs to provide real-world data. The main challenge was handling API rate limits and potential failures. I implemented caching and error handling to mitigate these issues.
3. User Interface: Streamlit was chosen for rapid prototyping and its ability to create interactive web applications with Python. The main challenge was designing an intuitive interface that showcased all features without overwhelming the user.
4. Data Management: Initially, I integrated the OpenTripMap API for more diverse and up-to-date destination information, but now I used the database due to API limitation and for better robustness. This required restructuring the data loading process and implementing efficient caching.
5. Performance Optimization: As the application grew, I faced performance issues, especially with embedding calculations. I implemented caching mechanisms to store embeddings and API results, significantly improving response times.
6. Testing and Refinement: Throughout the development process, I continuously tested the application, identifying and fixing bugs, and refining the user experience based on simulated user interactions.

The main challenges were balancing feature richness with MVP simplicity, managing API integrations, and ensuring good performance. I addressed these through careful feature selection, robust error handling, and performance optimizations.
