import pandas as pd
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
from langdetect import detect
import torch

# Set Pandas display options to show full text without truncation
pd.set_option('display.max_colwidth', None)  # This will show the full text in the columns

# Step 1: Fetch articles from News API
def fetch_articles(api_url, api_key):
    response = requests.get(f"{api_url}?apiKey={api_key}")
    if response.status_code != 200:
        print(f"API Request Failed: Status Code {response.status_code}")
        return pd.DataFrame()  # Return empty dataframe if request fails
    try:
        data = response.json()  # Get the JSON response
        print("API Response:", data.keys())  # Print the top-level keys to check the response structure
        articles = data.get('results', [])  # Articles are inside the 'results' key, not 'articles'
        if not articles:
            print("No articles found in the API response.")
        return pd.DataFrame(articles)
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        return pd.DataFrame()  # Return empty dataframe if JSON parsing fails

# Step 2: Filter only English articles based on title and description
def filter_english_articles(article_df):
    # Define a function to detect language and filter articles
    def is_english(text):
        try:
            return detect(text) == 'en'
        except:
            return False

    # Apply the language detection to title and description (or other text columns)
    article_df['is_english'] = article_df['title'].apply(is_english) | article_df['description'].apply(is_english)
    # Filter the dataframe to keep only English articles
    return article_df[article_df['is_english']]

# Step 3: Simulate user interactions (1 = interacted, 0 = not interacted)
def simulate_user_interactions(article_df):
    return np.random.choice([1, 0], size=(len(article_df),), p=[0.7, 0.3])

# Step 4: Extract BERT embeddings for articles
def extract_article_embeddings(article_df):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    embeddings = []

    for _, row in article_df.iterrows():
        title = str(row.get('title', ''))
        description = str(row.get('description', ''))
        content = str(row.get('content', '')) if 'content' in row else ''
        combined_text = f"{title} {description} {content}".strip()

        if not combined_text:
            embeddings.append(np.zeros(768))  # Return zero vector if no content
            continue

        inputs = tokenizer(combined_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            output = model(**inputs)
        embeddings.append(output.last_hidden_state.mean(dim=1).squeeze().numpy())

    return np.array(embeddings)

# Step 5: Create user profile based on interactions
def create_user_profile(user_interactions, article_embeddings):
    interacted_indices = np.where(user_interactions == 1)[0]
    if len(interacted_indices) == 0:
        print("No user interactions, using zero profile.")
        return np.zeros(article_embeddings.shape[1])  # Return a zero vector if no interactions
    return article_embeddings[interacted_indices].mean(axis=0)

# Step 6: Generate top recommendations using cosine similarity
def generate_recommendations(user_profile, article_embeddings, article_df, top_n=10):
    similarity_scores = cosine_similarity([user_profile], article_embeddings).flatten()
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]
    
    # Filter out rows with missing descriptions
    recommended_articles = article_df.iloc[top_indices]
    recommended_articles = recommended_articles[recommended_articles['description'].notna()]
    
    # Fallback for missing descriptions
    recommended_articles['description'] = recommended_articles['description'].apply(
        lambda x: x if pd.notna(x) else "Description not available"
    )
    
    # Remove duplicates based on title and description
    recommended_articles = recommended_articles.drop_duplicates(subset=['title', 'description'])

    return recommended_articles[['title', 'description']]

# Step 7: Print the recommendations in a readable table format
def print_recommendations(recommendations):
    print("\n Personalized News Feed:")
    print("-" * 80)
    for index, row in recommendations.iterrows():
        print(f"Title: {row['title']}\nDescription: {row['description']}\n")
        print("-" * 80)

# Main pipeline
def main():
    api_url = 'https://newsdata.io/api/1/news'
    api_key = 'pub_837292dc5985a12009412503c14621a59c91f'

    print("Fetching news articles...")
    article_df = fetch_articles(api_url, api_key)

    if article_df.empty:
        print("No articles fetched from API.")
        return

    print("Filtering English articles...")
    article_df = filter_english_articles(article_df)

    if article_df.empty:
        print(" No English articles found.")
        return

    print("Extracting BERT embeddings...")
    article_embeddings = extract_article_embeddings(article_df)

    if len(article_embeddings) == 0:
        print(" No embeddings created. Check article content.")
        return

    print("Simulating user interactions...")
    user_interactions = simulate_user_interactions(article_df)

    print("Creating user profile...")
    user_profile = create_user_profile(user_interactions, article_embeddings)

    print("Generating personalized recommendations...")
    personalized_feed = generate_recommendations(user_profile, article_embeddings, article_df)

    # Print the recommendations in a readable format
    print_recommendations(personalized_feed)

# Ensure the script runs when executed
if __name__ == "__main__":
    main()
