import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter
import nltk
import google.generativeai as genai
import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from googleapiclient.discovery import build
import urllib.parse as urlparse

# ---- Setup ----
nltk.download('stopwords', quiet=True)
load_dotenv()

# Replace with your actual API keys
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# ---- Core Logic Functions (from your original script) ----

def clean_comment(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()

analyzer = SentimentIntensityAnalyzer()

def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    return [word for word in words if word not in stop_words and len(word) > 2]

def sentiment_label(score):
    if score >= 0.3:
        return "positive"
    elif score <= -0.3:
        return "negative"
    else:
        return "neutral"

def build_summary(compound_score, keywords_list):
    label = sentiment_label(compound_score)
    summary = f"Overall, people are *{label}* about this video."
    if keywords_list:
        summary += f" Top trending topics in the comments include: {', '.join(keywords_list)}."
    return summary

def build_prompt(video_title, tags, sentiment_scores, top_keywords, top_comments):
    label = sentiment_label(sentiment_scores['compound'])
    keywords = ", ".join(top_keywords)

    positive_comments = "\n".join([f"- {c}" for c in top_comments['positive']])
    negative_comments = "\n".join([f"- {c}" for c in top_comments['negative']])
    
    return (
        f"Video Title: {video_title}\n"
        f"Video Tags: {tags}\n"
        f"Viewer Sentiment: {label} "
        f"(compound: {sentiment_scores['compound']:.2f}, pos: {sentiment_scores['pos']:.2f}, neu: {sentiment_scores['neu']:.2f}, neg: {sentiment_scores['neg']:.2f})\n"
        f"Top Keywords: {keywords}\n"
        f"\nTop Positive Comments:\n{positive_comments}\n"
        f"\nTop Negative Comments:\n{negative_comments}\n"
        f"\nWrite a detailed, insightful summary of viewer opinion for this video. "
        f"Analyze the provided sentiment scores, keywords, and sample comments. "
        f"Highlight any notable trends, concerns, or praise from viewers, and provide context where possible. "
        f"Do not simply repeat the information provided; synthesize it into an engaging and informative summary."
    )


def generate_gemini_summary(prompt):
    try:
        response = gemini_model.generate_content(prompt, generation_config={"max_output_tokens": 300})
        return response.text
    except Exception as e:
        print(f"Error generating Gemini summary: {e}")
        return "Error: Could not generate summary."

# ---- YouTube API & Analysis Function ----
def analyze_and_summarize_video(video_id):
    try:
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

        # Fetch video title and tags
        video_request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        video_response = video_request.execute()
        video_snippet = video_response['items'][0]['snippet']
        video_title = video_snippet['title']
        tags = ", ".join(video_snippet.get('tags', []))
        
        # Fetch comments
        comments = []
        comments_request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100
        )
        comments_response = comments_request.execute()
        
        for item in comments_response['items']:
            comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append({'comment_text': comment_text})

        if not comments:
            return {
                "video_title": video_title,
                "rule_based_summary": "No comments available for this video.",
                "gemini_summary": "No comments available for this video."
            }

        comments_df = pd.DataFrame(comments)
        comments_df['clean_comment'] = comments_df['comment_text'].apply(clean_comment)

        sentiment_scores_df = comments_df['clean_comment'].apply(analyzer.polarity_scores).apply(pd.Series)
        merged_df = pd.concat([comments_df, sentiment_scores_df], axis=1)

        # Get top comments for each sentiment
        top_comments = {
            'positive': merged_df.sort_values(by='compound', ascending=False)['clean_comment'].head(5).tolist(),
            'negative': merged_df.sort_values(by='compound')['clean_comment'].head(5).tolist(),
        }
        
        merged_df['keywords'] = merged_df['clean_comment'].apply(extract_keywords)

        # Calculate sentiment and get top keywords
        avg_sentiment = merged_df[['compound', 'pos', 'neu', 'neg']].mean().to_dict()
        all_keywords = [word for sublist in merged_df['keywords'] for word in sublist]
        top_keywords = [w for w, _ in Counter(all_keywords).most_common(5)]

        # Generate summaries
        rule_based_summary = build_summary(avg_sentiment['compound'], top_keywords)
        prompt = build_prompt(video_title, tags, avg_sentiment, top_keywords, top_comments)
        gemini_summary = generate_gemini_summary(prompt)
        
        return {
            "video_title": video_title,
            "rule_based_summary": rule_based_summary,
            "gemini_summary": gemini_summary
        }

    except Exception as e:
        print(f"An error occurred: {e}")
        return {
            "error": "Could not process video. Please check the URL and your API keys."
        }


# ---- API Route ----
@app.route('/summarize', methods=['POST'])
def summarize_video_api():
    data = request.json
    video_url = data.get('video_url')

    if not video_url:
        return jsonify({"error": "No video URL provided."}), 400

    try:
        url_data = urlparse.urlparse(video_url)
        query = urlparse.parse_qs(url_data.query)
        video_id = query.get('v')[0]
    except Exception:
        return jsonify({"error": "Invalid YouTube URL."}), 400

    summary_data = analyze_and_summarize_video(video_id)

    return jsonify(summary_data)

if __name__ == '__main__':
    # To run, use `flask run` in the terminal from the backend directory.
    # On Windows, you might need to use `set FLASK_APP=app.py` before `flask run`.
    app.run(debug=True)
