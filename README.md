# Dream Analyzer and Journal

This project is designed to analyze the dreams users experience using text input along with additional metadata such as sleep duration and sleep cycle. It utilizes various natural language processing (NLP) techniques and libraries to perform sentiment analysis, entity recognition, topic modeling, keyword extraction, and dream symbolism interpretation. Additionally, the project includes a dream journal feature to store and manage dream entries.

## Features

- **Dream Analysis**: Utilizes NLP techniques for sentiment analysis, entity recognition, topic modeling, and keyword extraction to analyze dream texts.
- **Sentiment Analysis**: Determines the sentiment (positive, negative, or neutral) of the dream text.
- **Entity Recognition**: Identifies entities (people, places, things) mentioned in the dream and provides interpretations.
- **Topic Modeling**: Identifies central themes or topics present in the dream text.
- **Keyword Extraction**: Extracts keywords from the dream text to highlight significant issues or feelings.
- **Dream Symbolism Interpretation**: Provides interpretations of dream entities based on common dream symbolism.
- **Dream Journal**: Allows users to store and manage their dream entries along with metadata such as sleep duration and sleep cycle.

## Technologies Used

- **Python**: Programming language used for backend development.
- **Flask**: Micro web framework used for building the web application.
- **Spacy**: Library used for natural language processing tasks such as entity recognition.
- **Gensim**: Library used for topic modeling.
- **Rake-NLTK**: Library used for keyword extraction.
- **WordCloud**: Library used for generating word clouds.
- **Matplotlib**: Library used for data visualization.
- **SQLite**: Embedded relational database used for storing dream entries.
- **VADER Sentiment Analysis**: Library used for sentiment analysis.
- **Requests**: Library used for making HTTP requests to fetch additional data.
- **Bootstrap**: Frontend framework used for styling.

## Usage

1. **Input Dream Text**: Enter the description of your dream along with additional metadata such as sleep duration and sleep cycle.
2. **View Analysis**: After submission, the system will analyze your dream and provide insights such as sentiment, entities, topics, keywords, and dream symbolism interpretations.
3. **Dream Journal**: You can view and manage your dream entries in the dream journal.

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/your-username/dream-analyzer-and-journal.git

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

3. Run the Flask application:

   ```bash
   python app.py

4. Access the application via http://localhost:5000 in your web browser.

Contributing
Contributions to the project are welcome! Feel free to open issues for bug fixes or feature requests, or submit pull requests for enhancements.

License
This project is licensed under the MIT License.

## Troubleshooting

### Error: Can't Find spaCy Model

If you encounter an error like `OSError: [E050] Can't find model 'en_core_web_md'`, follow these steps to resolve it:

1. Make sure you have installed spaCy and the required model by running:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_md
