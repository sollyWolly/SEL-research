import os
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

input_file = "data/CoP_Recording_3.11.21.txt"
output_file = "data/CoP_parsed_sentences.csv"

with open(input_file, 'r', encoding='utf-8', errors='ignore') as file:
    text = file.read()

# Sentence tokenization
sentences = sent_tokenize(text)

#Check if a sentence is too short to be its own independent sentence
def is_short_sentence(sentence, min_length=5):
    words = sentence.split()
    return len(words) < min_length

# Combine short sentences with the next sentence
processed_sentences = []
buffer = ""

for sentence in sentences:
    if is_short_sentence(sentence):
        buffer += " " + sentence.strip()
    else:
        if buffer:
            sentence = buffer + " " + sentence.strip()
            buffer = ""
        processed_sentences.append(sentence.strip())

df = pd.DataFrame({
    "Sentence ID": range(1, len(processed_sentences) + 1),
    "Sentence": processed_sentences
})

df.to_csv(output_file, index=False)

print(f"CSV file saved successfully at: {output_file}")
