from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ================== Load Dataset ==================
train_data = pd.read_csv('marketing_walmart_cleaned.csv', encoding='latin1')
train_data['Product Image Url'] = train_data['Product Image Url'].fillna('')

train_data['Product Rating'] = train_data['Product Rating'].fillna(0)
train_data['Product Reviews Count'] = train_data['Product Reviews Count'].fillna(0)
train_data['Product Tags'] = train_data['Product Tags'].fillna('')

# ================== TF-IDF ==================
tfidf = TfidfVectorizer(stop_words='english')

tfidf_matrix = tfidf.fit_transform(train_data['Product Tags'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ================== Recommendation Function ==================

def content_based_recommendations(product_name, top_n=12):

    matched_products = train_data[
        train_data['Product Name']
        .str.lower()
        .str.contains(product_name.lower(), na=False)
    ]

    if matched_products.empty:
        return pd.DataFrame()

    idx = matched_products.index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:top_n+1]

    product_indices = [i[0] for i in sim_scores]

    recommendations = train_data.iloc[product_indices]

    return recommendations[
        [
            'Product Name',
            'Product Brand',
            'Product Image Url',
            'Product Rating'
        ]
    ]

# ================== Routes ==================

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():

    product_name = request.form.get('product_name')

    recommendations = content_based_recommendations(product_name)

    if recommendations.empty:
        return render_template(
            'index.html',
            message="Product not found 😢"
        )

    return render_template(
        'index.html',
        recommendations=recommendations.to_dict('records'),
        searched_product=product_name
    )

# ================== Run ==================

if __name__ == '__main__':
    app.run(debug=True)