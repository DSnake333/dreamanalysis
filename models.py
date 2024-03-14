# models.py

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Dream(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(20))
    content = db.Column(db.Text)
    sentiment = db.Column(db.String(10))  # Add sentiment field
    entities = db.Column(db.Text)         # Add entities field
    topics = db.Column(db.Text)           # Add topics field
    keywords = db.Column(db.Text)         # Add keywords field

    def __init__(self, timestamp, content, sentiment=None, entities=None, topics=None, keywords=None):
        self.timestamp = timestamp
        self.content = content
        self.sentiment = sentiment
        self.entities = entities
        self.topics = topics
        self.keywords = keywords
