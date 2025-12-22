from flask import Flask, request, jsonify
from flask_cors import CORS
from googleapiclient.discovery import build
from textblob import TextBlob, Word
from collections import Counter
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from nltk.corpus import stopwords
import nltk
import numpy as np
import math
import re
import os
from dotenv import load_dotenv
from groq import Groq  # 👈 New Import

# ==============================
#      INITIAL SETUP
# ==============================
app = Flask(__name__)
CORS(app)
load_dotenv()

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
stop_words = set(stopwords.words("english"))

# Get Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not GROQ_API_KEY or not YOUTUBE_API_KEY:
    print("⚠️ Missing API keys. Please check your .env file.")
else:
    print("✅ API keys loaded successfully")

# ✅ Initialize Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)


# ==============================
#     TEXT CLEANING HELPERS
# ==============================
def clean_text_keep_punctuation(text: str) -> str:
    """Remove HTML, URLs, symbols; keep spacing."""
    text = BeautifulSoup(text, "html.parser").text
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^A-Za-z\s]", " ", text)
    return text.lower().strip()


def preprocess_text(text: str) -> str:
    """Clean + remove stopwords + lemmatize."""
    words = [
        str(Word(w).lemmatize())
        for w in clean_text_keep_punctuation(text).split()
        if w not in stop_words and len(w) > 2
    ]
    return " ".join(words)


# ==============================
#     FETCH YOUTUBE COMMENTS
# ==============================
def fetch_youtube_comments(video_url):
    """Fetch comments + like counts for weighting."""
    match = re.search(r"v=([A-Za-z0-9_-]{11})", video_url)
    if not match:
        raise ValueError("Invalid YouTube URL.")
    video_id = match.group(1)

    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments, next_page_token = [], None

    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            pageToken=next_page_token,
            textFormat="plainText",
        ).execute()

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            raw_text = snippet.get("textDisplay", "")
            like_count = snippet.get("likeCount", 0)
            preprocessed = preprocess_text(raw_text)
            if preprocessed:
                comments.append({
                    "raw_text": clean_text_keep_punctuation(raw_text),
                    "comment": preprocessed,
                    "like_count": like_count
                })

        next_page_token = response.get("nextPageToken")
        if not next_page_token or len(comments) >= 500:
            break

    title = youtube.videos().list(part="snippet", id=video_id).execute()["items"][0]["snippet"]["title"]
    print(f"📺 Fetched {len(comments)} comments for: {title}")
    return title, comments


# ==============================
#     SENTIMENT ANALYSIS
# ==============================
def analyze_sentiment(comments):
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    for c in comments:
        polarity = TextBlob(c["comment"]).sentiment.polarity
        if polarity > 0.1:
            sentiments["positive"] += 1
        elif polarity < -0.1:
            sentiments["negative"] += 1
        else:
            sentiments["neutral"] += 1
    return sentiments


# ==============================
#   PHRASE EXTRACTION (CHUNKING)
# ==============================
def extract_phrases_linguistic(text):
    """Extract meaningful multi-word phrases using POS tagging."""
    from nltk import pos_tag, RegexpParser, word_tokenize

    grammar = r"""
        NP: {<JJ.*>*<NN.*>+}    # Adjectives + Nouns
            {<NN.*><IN><NN.*>}  # Noun + Preposition + Noun
    """
    chunker = RegexpParser(grammar)
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    tree = chunker.parse(tagged)

    phrases = []
    for subtree in tree.subtrees(filter=lambda t: t.label() == 'NP'):
        phrase = " ".join(word for word, pos in subtree.leaves()).lower()
        if 2 <= len(phrase.split()) <= 5:
            phrases.append(phrase)
    return phrases


# ==============================
#   HYBRID WEIGHTED PHRASE MINING
# ==============================
def compute_weighted_phrases(comments, top_k=20, pmi_boost=0.6):
    phrase_weights = Counter()
    word_counts = Counter()
    total_words = 0

    for c in comments:
        text = c["raw_text"]
        phrases = extract_phrases_linguistic(text)
        sentiment = abs(TextBlob(c["comment"]).sentiment.polarity)
        likes = c.get("like_count", 0)
        engagement_weight = (1 + sentiment) * (1 + np.log1p(likes))

        for p in phrases:
            phrase_weights[p] += engagement_weight
            for w in p.split():
                word_counts[w] += 1
                total_words += 1

    # PMI-like phrase quality boost
    for p in list(phrase_weights.keys()):
        words = p.split()
        if len(words) > 1:
            pmi = 0
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                c_w1, c_w2 = word_counts[w1], word_counts[w2]
                joint = min(c_w1, c_w2)
                if c_w1 and c_w2:
                    pmi += math.log((joint * total_words) / (c_w1 * c_w2 + 1e-9))
            pmi = max(pmi, 0)
            phrase_weights[p] *= (1 + pmi_boost * pmi)

    total = sum(phrase_weights.values()) or 1.0
    normalized = {k: v / total for k, v in phrase_weights.items()}
    top_phrases = dict(sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:top_k])
    return top_phrases


# ==============================
#  SEMANTIC CLUSTERING
# ==============================
def cluster_similar_phrases(weighted_phrases, similarity_threshold=0.8):
    if not weighted_phrases:
        return {}

    phrases = list(weighted_phrases.keys())
    weights = np.array(list(weighted_phrases.values()))
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(phrases, convert_to_numpy=True)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - similarity_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    merged = {}
    for cluster_id in set(labels):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_phrases = [phrases[i] for i in cluster_indices]
        cluster_weights = [weights[i] for i in cluster_indices]
        rep_phrase = cluster_phrases[np.argmax(cluster_weights)]
        merged[rep_phrase] = {
            "score": sum(cluster_weights),
            "members": cluster_phrases
        }

    total = sum(v["score"] for v in merged.values()) or 1.0
    normalized = {
        k: {
            "score": v["score"] / total,
            "members": v["members"]
        }
        for k, v in merged.items()
    }

    top_clusters = dict(
        sorted(normalized.items(), key=lambda x: x[1]["score"], reverse=True)[:10]
    )
    return top_clusters


# ==============================
#    GROQ SUMMARIZATION
# ==============================
def generate_ai_summaries(video_title, comments, phrases):
    phrase_list = [f"{k} (score={v['score']:.3f})" for k, v in phrases.items()]

    full_prompt = f"""
You are an AI analyst studying audience reactions to the YouTube video: "{video_title}".

Key discussion phrases and weights:
{phrase_list}

Write a full-length insights report including:
- Main discussion themes
- Overall sentiment and tone
- Excitement and criticism areas
- Recommendations for the creator
Make it detailed and professional.
"""
    
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", # Highly capable, free tier model
            messages=[
                {"role": "system", "content": "You are a helpful YouTube analyst."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False,
            stop=None,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print("Groq API error:", e)
        return f"AI summary generation failed: {str(e)}"


# ==============================
#       MAIN ENDPOINT
# ==============================
@app.route("/summarize", methods=["POST"])
def summarize_video():
    try:
        data = request.get_json()
        video_url = data.get("video_url")
        if not video_url:
            return jsonify({"error": "YouTube URL is required"}), 400

        title, comments = fetch_youtube_comments(video_url)
        sentiments = analyze_sentiment(comments)
        weighted = compute_weighted_phrases(comments, top_k=25, pmi_boost=0.6)
        clustered = cluster_similar_phrases(weighted, similarity_threshold=0.8)

        rule_summary = f"The top audience discussion revolves around {', '.join(list(clustered.keys())[:5])}."
        
        # Calls Groq now
        full_summary = generate_ai_summaries(title, comments, clustered)
        short_summary = "Short summary not available for free version"

        return jsonify({
            "video_title": title,
            "rule_based_summary": rule_summary,
            "gemini_summary_full": full_summary,
            "gemini_summary_short": short_summary,
            "sentiment_data": sentiments,
            "top_keywords": clustered
        })

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500


# ==============================
#        RUN SERVER
# ==============================
if __name__ == "__main__":
    app.run(debug=True)