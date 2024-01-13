import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np
import preprocessing as pre


# import plotly.express as px 
# import plotly.graph_objects as go

st.set_page_config(page_title='Disaster Tweet Analysis', page_icon=':bar_chart:', layout='wide')

t1, t2 = st.columns((0.1, 1))

# t2.title("Disaster Tweet Analysis")
# t2.markdown("S")

def preprocessing_text(text):
    # Perform text cleaning
    cleaned_input_text = pre.clean_text(text[0], use_stemmer=True)
    return cleaned_input_text

def load_keras_model():
    model_path = "model_prep/model1.keras"  # Ganti dengan path model Keras Anda
    model = tf.keras.models.load_model(model_path,compile=False)
    return model

def vectorizer_and_load_model():
    with open("model_prep/count_vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    return vectorizer

def main():
    st.title("Deteksi Bencana dengan NLP - Keras Model")
    st.sidebar.title("Parameter Model")

    # Muat model dan vektorisasi teks
    keras_model = load_keras_model()
    vectorizer = vectorizer_and_load_model()

    # Tampilkan form untuk input teks
    user_input = st.text_input("Masukkan teks:", "")

    # Prediksi jika ada input
    if st.button("Prediksi"):
        clean_text = preprocessing_text([user_input])
        
        # Transformasi input menggunakan CountVectorizer
        user_input_vec = vectorizer.transform([clean_text])

        # Persiapkan input untuk model Keras
        input_array = user_input_vec.toarray()

        # Lakukan prediksi dengan model Keras
        prediction = keras_model.predict(input_array)[0][0]
        
        prediction_label = "Disaster" if prediction >= 0.5 else "Non-Disaster"

        st.success(f"Hasil Prediksi: {prediction_label} (Probabilitas: {prediction:.2f})")

if __name__ == "__main__":
    main()