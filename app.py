from flask import Flask, render_template, request, url_for, redirect
import spacy
# from spacy.pipeline import EntityRuler
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask import abort
from gensim import corpora, models
from rake_nltk import Rake
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from dream_data import dream_symbolism, patterns
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
# from flask_migrate import Migrate
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import json

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///dreams.db'
db = SQLAlchemy(app)

class Dream(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(20), nullable=False)
    sentiment = db.Column(db.String(10), nullable=False)
    entities = db.Column(db.String(200))
    topics = db.Column(db.String(200))
    keywords = db.Column(db.String(200))
    interpretation = db.Column(db.Text)
    dream_symbolism_result = db.Column(db.Text)
    user_guidance = db.Column(db.Text)
    detailed_sentiment_analysis = db.Column(db.Text)
    word_cloud_image_path = db.Column(db.String(100))  # Added field for word cloud image path
    emotion_distribution_chart_path = db.Column(db.String(100))  # Added field for emotion distribution pie chart path
    user_input = db.Column(db.Text)
    sleep_duration = db.Column(db.Integer)  # Added field for sleep duration
    sleep_quality = db.Column(db.Integer)
    sleep_stage = db.Column(db.String(20))


nlp = spacy.load("en_core_web_md")
ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
ruler.add_patterns(patterns)

def analyze_sentiment(text):
    sentences = re.split(r'[.!?]', text)
    analyzer = SentimentIntensityAnalyzer()

    sentiments = []
    for idx, sentence in enumerate(sentences):
        sentence = sentence.strip()  # Remove leading/trailing whitespaces
        if not sentence:
            continue  # Skip empty sentences

        sentiment_score = analyzer.polarity_scores(sentence)["compound"]

        if sentiment_score >= 0.05:
            sentiment = "Positive"
        elif sentiment_score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        sentiments.append(sentiment)
        print(f"Sentence {idx + 1}: {sentence} - Sentiment: {sentiment}")

    return sentiments

def extract_entities(text):
    doc = nlp(text)
    entities = [ent.text for ent in doc.ents]
    return entities

def perform_topic_modeling(text):
    tokens = [token.text for token in nlp(text)]
    dictionary = corpora.Dictionary([tokens])
    corpus = [dictionary.doc2bow(tokens)]
    lda_model = models.LdaModel(corpus, num_topics=1, id2word=dictionary)
    topics = lda_model.get_document_topics(corpus[0])
    return topics

def extract_keywords(text):
    r = Rake()
    r.extract_keywords_from_text(text)
    keywords = r.get_ranked_phrases()
    return keywords

def provide_interpretation(sentiment, entities, topics, keywords):
    interpretation = "\nInterpretation:\n"

    # Add sentiment interpretation
    interpretation += f"Sentiment: {sentiment}\n"
    # Add entities interpretation
    if entities:
        interpretation += "\nEntities in your dream:\n"
        for entity in entities:
            interpretation += f"- {entity} may hold significance in your subconscious thoughts.\n"
    # Add topics interpretation
    if topics:
        interpretation += "\nTopics identified in your dream:\n"
        for topic, score in topics:
            interpretation += f"- Topic {topic + 1}: {score:.2f} - Consider the theme of this topic in relation to your experiences.\n"
    # Add keywords interpretation
    if keywords:
        interpretation += "\nKeywords extracted from your dream:\n"
        for keyword in keywords:
            interpretation += f"- {keyword} could be key elements in your dream's narrative.\n"

    return interpretation

def provide_dream_symbolism(input_text, dream_symbolism):
    if isinstance(input_text, str):
        text = input_text
    elif isinstance(input_text, list):
        # Join the list of strings into a single string
        text = ' '.join(input_text)
    else:
        raise ValueError("Input must be a string or a list of strings")

    dream_symbolism_result = "\n \n"
    unique_entities = set()  # Keep track of unique entities

    # Extract entities using spaCy NER
    doc = nlp(text)
    entities = [ent.text.lower() for ent in doc.ents]

    # Iterate through recognized entities
    for entity in entities:
        # Check if the entity is present in the dream symbolism dictionary
        if entity in dream_symbolism and entity not in unique_entities:
            dream_symbolism_result += f"{entity}: {dream_symbolism[entity]}\n"
            unique_entities.add(entity)

    if not unique_entities:
        dream_symbolism_result += "No entities found in the dream."

    return dream_symbolism_result


def provide_user_guidance():
    user_guidance = "\nUser Guidance:\n"
    user_guidance += "1. Sentiment: Positive dreams generally reflect positive emotions, while negative dreams may highlight concerns.\n"
    user_guidance += "2. Entities: Pay attention to named entities; they may represent key aspects of your subconscious thoughts.\n"
    user_guidance += "3. Topics: Explore the main themes identified; they provide insights into the core elements of your dream.\n"
    user_guidance += "4. Keywords: Focus on extracted keywords; they could be pivotal elements in understanding your dream.\n"
    user_guidance += "5. Dream Symbolism: Check the interpretation of specific entities based on common dream symbolism.\n"
    return user_guidance

def analyze_sentiment_advanced(text):
    # Use VADER SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)

    # Additional rule-based analysis based on patterns in the text
    if re.search(r"\b(?:good|excellent|positive)\b", text, flags=re.IGNORECASE):
        sentiment_scores["compound"] += 0.2  # Increase the compound score for positive keywords
    elif re.search(r"\b(?:bad|negative)\b", text, flags=re.IGNORECASE):
        sentiment_scores["compound"] -= 0.2  # Decrease the compound score for negative keywords

    # Adjust the sentiment label based on the compound score
    sentiment_label = "Positive" if sentiment_scores["compound"] >= 0.05 else "Negative" if sentiment_scores["compound"] <= -0.05 else "Neutral"

    return sentiment_label, sentiment_scores

def analyze_sentiment_detailed(text):
    sentences = [sentence.text for sentence in nlp(text).sents]
    detailed_analysis = {}

    analyzer = SentimentIntensityAnalyzer()

    for idx, sentence in enumerate(sentences):
        sentiment_score = analyzer.polarity_scores(sentence)["compound"]

        # Create a nested dictionary to store sentiment information
        sentiment_info = {"original_sentiment": sentiment_score}

        # Include the enhanced sentiment label in the detailed analysis
        enhanced_sentiment_label, _ = analyze_sentiment_advanced(sentence)
        sentiment_info["enhanced_sentiment"] = enhanced_sentiment_label

        # Assign the nested dictionary to the sentence key
        detailed_analysis[sentence] = sentiment_info

    return detailed_analysis

def generate_word_cloud_image(text, dream_id):
    # Customize the WordCloud parameters
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        contour_color='steelblue',  # Outline color
        contour_width=2,            # Outline width
        max_words=200,              # Maximum number of words in the cloud
        colormap='viridis',         # Color map
        font_path='RadiantKingdom-mL5eV.ttf',  # Specify the path to your custom font
        stopwords=set(STOPWORDS),   # Set of stopwords
        random_state=42
    ).generate(text)

    # Plot the WordCloud image
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    # Save word cloud image to a file or convert to binary data
    image_path = f"wordclouds/{dream_id}_wordcloud.png"
    plt.savefig(f"static/{image_path}")  # Save the image in the 'static' folder
    plt.close()

    return image_path

def generate_emotion_distribution_pie_chart(sentiments, dream_id, chart_type='pie'):
    # Count occurrences of each sentiment
    sentiment_counts = {'Positive': 0, 'Neutral': 0, 'Negative': 0}
    for sentiment in sentiments:
        sentiment_counts[sentiment] += 1

    # Define colors for each sentiment
    colors = {'Positive': 'green', 'Neutral': 'gray', 'Negative': 'red'}

    # Plot the pie chart
    plt.pie(sentiment_counts.values(), labels=sentiment_counts.keys(), autopct='%1.1f%%', colors=[colors[s] for s in sentiment_counts.keys()])

    # Add title
    plt.title('Sentiment Distribution')

    # Save the pie chart image
    image_path = f"emotion_distribution_pie_chart_{dream_id}.png"
    plt.savefig(f"static/{image_path}")
    plt.close()

    return image_path

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/dream_journal")
def dream_journal():
    # Retrieve dream entries from the database
    dream_journal = Dream.query.all()

    return render_template("dream_journal.html", dream_journal=dream_journal)

@app.route("/dream/delete/<int:index>", methods=["POST"])
def delete_dream(index):
    dream_entry = Dream.query.get(index + 1)  # Adjust index to match 1-based indexing
    if dream_entry:
        db.session.delete(dream_entry)
        db.session.commit()
        return redirect(url_for("dream_journal"))
    else:
        abort(404, "Dream entry not found.")

@app.route("/index", methods=["GET"])
def index():
    # Retrieve dream entries from the database
    dream_journal = Dream.query.all()
    return render_template("index.html", dream_journal=dream_journal, show_delete_button=True)

@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    dream_entry = None  # Initialize dream_entry variable

    user_input = request.form["user_input"]
    sleep_duration = int(request.form["sleep_duration"])
    sleep_quality = int(request.form["sleep_quality"])
    sleep_stage = request.form["sleep_stage"]

    # Sentiment Analysis
    sentiment = analyze_sentiment(user_input)

    # Named Entity Recognition
    entities = extract_entities(user_input)

    # Topic Modeling
    topics = perform_topic_modeling(user_input)

    # Keyword Extraction
    keywords = extract_keywords(user_input)

    sentiment_label, _ = analyze_sentiment_advanced(user_input)

    # Provide Interpretation
    interpretation = provide_interpretation(sentiment, entities, topics, keywords)

    # Incorporate Dream Symbolism
    dream_symbolism_result = provide_dream_symbolism(entities, dream_symbolism)

    # Enhance User Guidance
    user_guidance = provide_user_guidance()

    # Generate detailed sentiment analysis
    detailed_sentiment_analysis = analyze_sentiment_detailed(user_input)

    # Convert detailed sentiment analysis to JSON
    detailed_sentiment_analysis_json = json.dumps(detailed_sentiment_analysis)

    # Create and save the Dream entry to the database
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dream_entry = Dream(
        timestamp=timestamp,
        sentiment=sentiment_label,
        entities=', '.join(entities),
        topics=', '.join([str(topic) for topic, _ in topics]),
        keywords=', '.join(keywords),
        interpretation=interpretation,
        dream_symbolism_result=dream_symbolism_result,
        user_guidance=user_guidance,
        detailed_sentiment_analysis=detailed_sentiment_analysis_json,
        word_cloud_image_path=None,  # Initialize with None, will be updated later
        emotion_distribution_chart_path=None,  # Initialize with None, as it will be updated later
        user_input=user_input,
        sleep_duration=sleep_duration,
        sleep_quality=sleep_quality,
        sleep_stage=sleep_stage,  # New line to store sleep stage
    )

    # Add dream entry to the session
    db.session.add(dream_entry)
    db.session.commit()

    # Generate word cloud image after dream_entry.id is available
    word_cloud_image_path = generate_word_cloud_image(user_input, dream_entry.id)

    # Update the word_cloud_image_path in the database
    dream_entry.word_cloud_image_path = f"wordclouds/{dream_entry.id}_wordcloud.png"
    db.session.add(dream_entry)
    db.session.commit()

    # Generate emotion distribution pie chart
    emotion_distribution_pie_chart_path = generate_emotion_distribution_pie_chart(sentiment, dream_entry.id)
    dream_entry.emotion_distribution_chart_path = emotion_distribution_pie_chart_path
    db.session.commit()

    # Redirect to the results page
    return redirect(url_for("results",
                            sentiment=sentiment_label,
                            entities=entities,
                            topics=topics,
                            keywords=keywords,
                            interpretation=interpretation,
                            dream_symbolism_result=dream_symbolism_result,
                            user_guidance=user_guidance,
                            detailed_sentiment_analysis=detailed_sentiment_analysis_json,
                            word_cloud_image_path=word_cloud_image_path,
                            emotion_distribution_pie_chart_path=emotion_distribution_pie_chart_path,
                            show_word_cloud=True,  # Always show word cloud
                            sleep_duration=sleep_duration,
                            sleep_quality=sleep_quality,
                            sleep_stage=sleep_stage
                            ))
@app.route("/results")
def results():
    emotion_distribution_pie_chart_path = request.args.get('emotion_distribution_pie_chart_path')
    print("Emotion Distribution Pie Chart Path:", emotion_distribution_pie_chart_path)

    return render_template(
        "results.html",
        sentiment=request.args.get('sentiment'),
        entities=request.args.get('entities'),
        topics=request.args.get('topics'),
        keywords=request.args.get('keywords'),
        interpretation=request.args.get('interpretation'),
        dream_symbolism_result=request.args.get('dream_symbolism_result'),
        user_guidance=request.args.get('user_guidance'),
        detailed_sentiment_analysis=request.args.get('detailed_sentiment_analysis'),
        word_cloud_image_path=request.args.get('word_cloud_image_path'),
        emotion_distribution_pie_chart_path=emotion_distribution_pie_chart_path,
        show_word_cloud=bool(request.args.get('show_word_cloud')),  # Pass boolean variable
        sleep_duration=request.args.get('sleep_duration'),
        sleep_quality=request.args.get('sleep_quality'),
        sleep_stage=request.args.get('sleep_stage')
    )

@app.route("/dream/<int:index>")
def view_dream(index):
    index += 1
    dream_entry = Dream.query.get(index)
    if dream_entry:
        try:
            # Attempt to convert detailed_sentiment_analysis to a dictionary
            detailed_sentiment_analysis = (
                json.loads(dream_entry.detailed_sentiment_analysis)
                if dream_entry.detailed_sentiment_analysis
                else {}
            )
        except json.JSONDecodeError:
            # Handle the case where the string is not valid JSON
            detailed_sentiment_analysis = {}

        # Add the print statement here
        print("Word Cloud Image Path:", dream_entry.word_cloud_image_path)

        # Check if emotion_distribution_pie_chart_path is not empty
        if dream_entry.emotion_distribution_chart_path:
            emotion_distribution_pie_chart_path = dream_entry.emotion_distribution_chart_path
        else:
            emotion_distribution_pie_chart_path = None

        return render_template(
            "results.html",
            sentiment=dream_entry.sentiment,
            interpretation=dream_entry.interpretation,
            dream_symbolism_result=dream_entry.dream_symbolism_result,
            user_guidance=dream_entry.user_guidance,
            detailed_sentiment_analysis=detailed_sentiment_analysis,
            word_cloud_image_path=dream_entry.word_cloud_image_path,
            emotion_distribution_pie_chart_path=emotion_distribution_pie_chart_path,
            sleep_duration=dream_entry.sleep_duration,
            sleep_quality=dream_entry.sleep_quality,
            sleep_stage=dream_entry.sleep_stage
        )
    else:
        return "Dream entry not found."

if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create new tables
    app.run(debug=True)

@app.teardown_appcontext
def shutdown_session(exception=None):
    db.session.remove()
