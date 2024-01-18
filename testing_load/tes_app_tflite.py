import streamlit as st
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np
import prep_app.preprocessing as pre
import spacy
import spacy_streamlit as ss

st.set_page_config(page_title='Disaster Tweet Analysis', page_icon=':bar_chart:', layout='wide')

def preprocessing_text(text):
    # Perform text cleaning
    cleaned_input_text = pre.clean_text(text[0], use_stemmer=True)
    return cleaned_input_text

def vectorizer_and_load_model():
    with open("model_prep/count_vectorizer.pkl", "rb") as vectorizer_file: 
        vectorizer = pickle.load(vectorizer_file)
        
    tflite_model_path = 'model_prep/quantized_model1.tflite'
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    return vectorizer, interpreter

def extract_entities(text):
    spacy_model = spacy.load('en_core_web_sm')
    doc = spacy_model(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities 

def main():
    st.title("Deteksi Bencana dengan NLP - TensorFlow Lite")
    st.sidebar.title("Parameter Model")

    # Muat model dan vektorisasi teks
    vectorizer, interpreter = vectorizer_and_load_model()

    # Tampilkan form untuk input teks
    user_input = st.text_input("Masukkan teks:", "")

    # Prediksi jika ada input
    if st.button("Prediksi"):
        clean_text = preprocessing_text([user_input])
        
        # Transformasi input menggunakan CountVectorizer
        user_input_vec = vectorizer.transform([clean_text])

        # Persiapkan input untuk model TensorFlow Lite
        input_tensor_index = interpreter.get_input_details()[0]['index']
        interpreter.set_tensor(input_tensor_index, np.array(user_input_vec.toarray(), dtype=np.float32))

        # Lakukan inferensi
        interpreter.invoke()

        # Ambil output dari model TensorFlow Lite
        output_tensor_index = interpreter.get_output_details()[0]['index']
        prediction = interpreter.get_tensor(output_tensor_index)[0][0]
        
        prediction_label = "Disaster" if prediction >= 0.5 else "Non-Disaster"

        # Ekstraksi entitas dari user_input
        entities = extract_entities(user_input)
        
        
        st.success(f"Hasil Prediksi: {prediction_label} (Probabilitas: {prediction:.2f})")
        
        if entities:
            st.write("Entitas yang terdeteksi:")
            for ent, label in entities:
                st.write(f"- {ent} ({label})")
        
        

if __name__ == "__main__":
    main()