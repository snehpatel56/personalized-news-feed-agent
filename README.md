
# Personalized News Feed Agent

This Python project fetches the latest news articles from the [NewsData API](https://newsdata.io/), filters English-language content, simulates user interactions, and recommends articles using BERT-based embeddings and cosine similarity.

---

##  Features

- Fetches real-time news using NewsData API
- Filters only English-language articles
- Uses BERT to extract article embeddings
- Simulates user preferences (interacted vs. not interacted)
- Recommends top news articles based on user profile similarity


## make Virutal Environment using this

python -m venv venv

## Active  the Virtual Environment

.\venv\Scripts\activate



---

##  Requirements

Install all required packages using:

```bash
pip install -r requirements.txt
```

---

##  API Key

Get a free API key from [https://newsdata.io/](https://newsdata.io/).

Update the `api_key` variable in the script:


api_key = 'YOUR_API_KEY_HERE'


---

##  How to Run

1. Clone the repository or copy the script to your local machine.
2. Install the dependencies.
3. Run the script:


python main.py

or

## Deploy it on streamlit 
push code to github and then deploy easily



Output

A list of  personalized news articles based on simulated user preferences, printed in a readable table format.

---

##  Model

- Uses `bert-base-uncased` from Hugging Face Transformers
- Embeddings extracted using the mean of BERT's final hidden state
- Similarity calculated using `cosine_similarity` from `scikit-learn`

---

##  Notes

- Make sure you have a stable internet connection when downloading the BERT model or accessing the API.
- This script simulates interactions randomly. You can replace it with actual user preference data.

---


