from flask import Flask, request, jsonify
import tensorflow_hub as hub
import numpy as np
import pandas as pd
from flask_cors import CORS

# Load Universal Sentence Encoder module
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print("Module %s loaded" % module_url)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Function to embed input
def embed(input):
    return model(input)

def get_similarity(sentence1, sentence2):
    # Embed the input sentences
    embedding1 = embed([sentence1])[0]
    embedding2 = embed([sentence2])[0]
    # Compute the cosine similarity between the embeddings
    similarity = np.inner(embedding1, embedding2)
    return similarity * 100  # Convert cosine similarity to percentage

# Function to calculate similarity between tags in CSV and selected book
def calculate_tag_similarity(selected_book_tags, csv_tags):
    similarity_list = []
    for tag_title, tag_text in csv_tags.items():
        similarity = get_similarity(selected_book_tags, tag_text)
        similarity_list.append({"title": tag_title, "similarity_percentage": similarity})
    # Sort the similarity list by similarity percentage in descending order
    similarity_list = sorted(similarity_list, key=lambda x: x["similarity_percentage"], reverse=True)
    # Return the top 50 similar tags
    return similarity_list[:50]

# Load CSV file
csv_file_path = "books__with-tags.csv"  # Provide the path to your CSV file
data = pd.read_csv(csv_file_path)

@app.route('/api/get_similar_tags', methods=['POST'])
def get_similar_tags():
    # Get input JSON data
    input_data = request.json
    selected_book_tags = input_data[0]["tags"]

    # Extract tags along with their titles from the CSV file
    csv_tags = {}
    for index, row in data.iterrows():
        title = row["title"]
        tags = row["tags"]
        csv_tags[title] = tags

    # Calculate similarity between tags in CSV and selected book
    similar_tags = calculate_tag_similarity(selected_book_tags, csv_tags)

    # Return response
    return jsonify(similar_tags)

if __name__ == '__main__':
    app.run()
