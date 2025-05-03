import os
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# Ensure necessary NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')

# Additional stopwords (filler words)
additional_stopwords = {"um", "im", "uh", "like", "know", "going", "thats", "youre",
                        "get", "really", "theyre", "us", "theres", "gon", "around",
                        "dont", "want", "one", "think", "well", "see", "way", "na", "go"}

# Load the Llama output CSV
llama_file = 'llama_output/llama_output_20250304_164331.csv'
df = pd.read_csv(llama_file)

# Ensure the dataframe has necessary columns
if 'Sentence' not in df.columns or 'Self Care' not in df.columns:
    raise ValueError("CSV must contain 'Sentence' and 'Self Care' columns")

# Split sentences into 'yes' and 'no'
df_yes = df[df['Self Care'].str.lower() == 'yes']
df_no = df[df['Self Care'].str.lower() == 'no']

# Function to map POS tags to WordNet format for lemmatization
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # Default to noun

def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\b(?:redacted)\b', '', text)
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    
    # Tokenize
    tokens = word_tokenize(text)

    # POS tagging
    tagged_tokens = pos_tag(tokens)

    # Define valid POS tags to keep
    valid_tags = {'NN', 'NNP', 'JJ', 'RB', 'VB', 'CD'}
    
    # Initialize lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Stopwords removal
    stop_words = set(stopwords.words('english')).union(additional_stopwords)

    # Lemmatization and filtering based on POS tagging
    filtered_tokens = [
        lemmatizer.lemmatize(word, get_wordnet_pos(tag))
        for word, tag in tagged_tokens
        if tag in valid_tags and word not in stop_words
    ]
    
    return ' '.join(filtered_tokens)

# Apply preprocessing
df_yes['Processed_Sentence'] = df_yes['Sentence']#.apply(preprocess)
df_no['Processed_Sentence'] = df_no['Sentence']#.apply(preprocess)

# Function to compute BoW and TF-IDF
def compute_bow_tfidf(df, label):
    sentences = df['Processed_Sentence'].dropna().tolist()
    
    # Bag of Words with unigrams and bigrams
    count_vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 2))
    count_matrix = count_vectorizer.fit_transform(sentences)
    bow_words = count_vectorizer.get_feature_names_out()
    bow_values = count_matrix.toarray().sum(axis=0)
    
    # TF-IDF with unigrams and bigrams
    tfidf_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    tfidf_words = tfidf_vectorizer.get_feature_names_out()
    tfidf_values = tfidf_matrix.toarray().sum(axis=0)
    
    # Save to CSV sorted by weight
    bow_df = pd.DataFrame({'Word': bow_words, 'Weight': bow_values}).sort_values(by='Weight', ascending=False)
    tfidf_df = pd.DataFrame({'Word': tfidf_words, 'Weight': tfidf_values}).sort_values(by='Weight', ascending=False)
    
    bow_df.to_csv(f'models_output/bow_{label}.csv', index=False)
    tfidf_df.to_csv(f'models_output/tfidf_{label}.csv', index=False)
    
# Compute and save for both 'yes' and 'no'
compute_bow_tfidf(df_yes, 'yes')
compute_bow_tfidf(df_no, 'no')

print("BoW and TF-IDF calculations completed. Results saved in bow_yes.csv, bow_no.csv, tfidf_yes.csv, and tfidf_no.csv, sorted by weight.")
