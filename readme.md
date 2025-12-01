# Disaster Tweets Prediction and Location Identification using NLP 
---
### Application Demo 🎥  

A short demo of the application interface can be viewed here:
🔗 [Application Demo](https://youtu.be/RAt05cyo4mQ?si=uXKsuxK5iZOIZz_o)

## Background 👨‍💻

In the digital era, social media platforms have evolved into dynamic spaces where individuals share thoughts, experiences, and real-time information during emergencies and disasters. Among these platforms, **Twitter serves as a valuable source of crowdsourced data** that, if effectively leveraged, can significantly support emergency response, disaster management, humanitarian operations, and news media analysis.

During disasters, Twitter becomes a crucial communication channel where affected individuals share firsthand reports, request assistance, and disseminate critical information. In such situations, the swift and accurate identification of disaster-related tweets becomes essential. By categorizing tweets into *disaster-related* and *not disaster-related*, we can enable faster and more precise response for both natural and human-caused emergencies.

**Natural disasters** encompass catastrophic events triggered by natural forces—such as tornadoes, hurricanes, floods, wildfires, earthquakes, and droughts—which pose serious threats to life and infrastructure.
**Human-caused disasters**, on the other hand, stem from intentional or negligent actions, such as industrial accidents, acts of terrorism, shootings, and mass violence, often resulting in trauma and major societal disruption.

Additionally, extracting geographic information from tweets enables **location-based disaster analysis**, supporting humanitarian organizations in identifying affected regions and allocating resources more efficiently.

---

## Problem Statement 📝

The core challenge is to develop a robust Natural Language Processing (NLP) model capable of identifying disaster-related content within millions of noisy, informal, and context-heavy tweets. Specific challenges include:

* **Noisy & Informal Language** — Tweets frequently use abbreviations, slang, emojis, and non-standard writing.
* **Contextual Ambiguity** — Tweets may imply a disaster scenario indirectly without explicit keywords.
* **Named Entity Extraction** — Identifying relevant entities (locations, organizations, event names) in noisy text is critical for context understanding.

---

## Objectives 📚

This research aims to build a scalable and reliable classification model to support faster disaster identification and response. Key objectives include:

* **Enhancing Model Robustness**
  Ensuring strong model performance across diverse linguistic styles and disaster scenarios.

* **Multilingual Support**
  Expanding model capabilities to analyze tweets in multiple languages for global applicability.

* **Geographical Analysis**
  Detecting disaster-impacted regions using location-based information extracted from tweets.

---

## Notebook Link 📎

You can directly explore the notebook, experiments, and analysis here:
🔗 [https://www.kaggle.com/code/sardiirfansyah/nlp-eda-ner-glove-lstm-gru-cnn-tflite](https://www.kaggle.com/code/sardiirfansyah/nlp-eda-ner-glove-lstm-gru-cnn-tflite)

---
## Tech Stack 🧠🛠️

The technologies and approaches used in this project include:

* **Python**
* **Natural Language Processing (NLP) & Text Preprocessing**
* **Named Entity Recognition (NER)**
* **Word Embeddings (GloVe)**
* **Deep Learning Models:**

  * LSTM
  * GRU
  * CNN for text
* **TensorFlow Lite (TFLite) for lightweight on-device deployment**
* **Streamlit for model inference UI and interactive demonstration**
* **Kaggle Notebooks for experimentation and model development**
* **Matplotlib / Seaborn for exploratory analysis and visualization**

---


