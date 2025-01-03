from flask import Flask, render_template, request
import nltk
import re
import stanza
from nltk.corpus import stopwords as s_w
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import advertools as adv
import os

app = Flask(__name__)

# Setup for languages
languages = {'hindi': 'hi', 'tamil': 'ta', 'telugu': 'te', 'urdu': 'ur'}
languages_code = {1: 'English', 2: 'Telugu', 3: 'Hindi', 4: 'Tamil', 5: 'Urdu'}
for lang_code in languages.values():
    stanza.download(lang_code)
nlp_pipelines = {lang: stanza.Pipeline(code) for lang, code in languages.items()}

language_suffixes = {
    "telugu": ["గా", "ను", "కి", "లో", "మీద"],
    "hindi": ["ने", "ता", "ही", "से", "को"],
    "tamil": ["ஆன்", "இன்", "உம்", "க்கு", "ல்"],
    "urdu": ["نے", "گا", "گی", "کا", "کی"],
}

# Functions
def tokenize(text):
    return nltk.word_tokenize(text)

def change_case(text, language):
    if language == 'english':
        return text.lower(), text.upper()
    else:
        return f"Case change not supported for {language.upper()}."
    
def remove_punctuations(text):
    return re.sub(r'[^\u0C00-\u0C7F\u0900-\u097F\u0600-\u06FF\u0B80-\u0BFFa-zA-Z\s]', '', text)

def remove_stopwords(text, language):
    words = text.split()
    if language == 'english':
        stop_words = s_w.words('english')
    else:
        stop_words = sorted(adv.stopwords[language])
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def stemming(text, language):
    words = text.split()
    if language == 'english':
        stemmer = PorterStemmer()
        return [stemmer.stem(word) for word in words]
    return [stem_language_word(word, language) for word in words]

def stem_language_word(word, language):
    suffixes = language_suffixes.get(language.lower(), [])
    for suffix in suffixes:
        if word.endswith(suffix):
            return word[:-len(suffix)]
    return word

def lemmatization(text, language):
    if language == 'english':
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        return [lemmatizer.lemmatize(word) for word in words]
    doc = nlp_pipelines[language](text)
    return [word.lemma for sent in doc.sentences for word in sent.words]

# Routes
@app.route('/')
def index():
    return render_template('index.html', languages=languages_code)

@app.route('/process', methods=['POST'])
def process():
    language_num = int(request.form['language'])
    text = request.form['text']
    language = languages_code[language_num]
    
    if language not in languages_code.values():
        return "Invalid language selection."
    
    language = language.lower()
    
    processed_data = {
        "tokenized": tokenize(text),
        "case_changed": change_case(text, language),
        "punctuation_removed": remove_punctuations(text),
        "stopwords_removed": remove_stopwords(text, language),
        "stemming": stemming(text, language),
        "lemmatization": lemmatization(text, language),
    }
    return render_template('result.html', data=processed_data)

if __name__ == '__main__':
    nltk.download('punkt')
    nltk.download('stopwords')
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
