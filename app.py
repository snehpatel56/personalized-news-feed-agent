# app.py

import streamlit as st
from main import (
    fetch_articles,
    filter_english_articles,
    extract_article_embeddings,
    simulate_user_interactions,
    create_user_profile,
    generate_recommendations
)

st.set_page_config(page_title="News Feed Agent", layout="wide")
st.title("Personalized News Feed Agent")

# User inputs (Optional enhancements later)
api_url = 'https://newsdata.io/api/1/news'
api_key = st.secrets["pub_837292dc5985a12009412503c14621a59c91f"]  # Securely load from secrets

if st.button("Generate My News Feed"):
    with st.spinner("Fetching news articles..."):
        article_df = fetch_articles(api_url, api_key)

    if article_df.empty:
        st.error("No articles fetched.")
    else:
        with st.spinner("Filtering English articles..."):
            article_df = filter_english_articles(article_df)

        with st.spinner("Extracting article embeddings..."):
            article_embeddings = extract_article_embeddings(article_df)

        with st.spinner("Simulating user preferences..."):
            user_interactions = simulate_user_interactions(article_df)
            user_profile = create_user_profile(user_interactions, article_embeddings)

        with st.spinner("Generating recommendations..."):
            recommendations = generate_recommendations(user_profile, article_embeddings, article_df)

        st.success("Here are your personalized news recommendations:")
        for _, row in recommendations.iterrows():
            st.markdown(f"### {row['title']}")
            st.write(row['description'])
            st.markdown("---")
