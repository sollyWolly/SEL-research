# SEL-Detection-NLP

[![Symposium Poster](./data/symposium_poster.png)](./data/symposium_poster.png)

This repository contains the code and research pipeline for my undergraduate research at the SHIFT Research Institute at UC Berkeley. The project explores how Natural Language Processing (NLP) and AI-based methods can be used to identify evidence of Social and Emotional Learning (SEL) within educator conversations.

## About the Project

In partnership with California’s Community of Practice (CoP) initiative, this study investigates whether SEL-related themes—such as self-care, emotional regulation, and psychological resilience—can be detected in educator-led conversations using AI-driven language models.

The core research question:  
Can AI and NLP help surface meaningful emotional learning signals in real-world classroom and educator dialogue?

## Data and Confidentiality

The transcripts used in this study originate from professional development meetings and trainings conducted by educators, facilitators, and local education officials across California. These conversations often include reflections on trauma-informed practices and mental health in educational settings.

Because these transcripts involve real individuals and potentially sensitive discussions, all data is treated as confidential. Personally identifiable information (PII) has been removed, and **no raw transcript data is included in this repository**. Only anonymized, aggregate outputs (such as term frequency rankings and evaluation metrics) and supporting code are provided here.

All processing, labeling, and evaluation steps respect the privacy of those involved and comply with ethical research standards.

## Methods

This study uses a combination of:
- Large Language Model classification (via [Ollama](https://ollama.com/))
- Contextual prompt engineering using sentence windowing
- Preprocessing using tokenization, lemmatization, and stopword removal
- TF-IDF and Bag of Words models for keyword extraction and comparison
- Evaluation via confusion matrices, precision, recall, F1 score, and specificity

The `AM Updated Rating` column—manually annotated by human reviewers—serves as ground truth for model validation.

## Author

**Solomon Cheung**  
Undergraduate Researcher, Computer Science
UC Berkeley – SHIFT Research Institute
Email: [solc@berkeley.edu](mailto:solc@berkeley.edu)