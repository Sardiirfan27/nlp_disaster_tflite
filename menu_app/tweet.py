import streamlit as st
import tensorflow as tf
import pickle
import hydralit_components as hc
import numpy as np
import prep_app.preprocessing as pre
from prep_app.translator import detect_language, translate_to_english
from prep_app.ner import perform_ner

def preprocessing_text(text):
    # Perform text cleaning
    cleaned_input_text = pre.clean_text(text, use_stemmer=True)
    return cleaned_input_text

def vectorizer_and_load_model():
    with open("model_prep/count_vectorizer.pkl", "rb") as vectorizer_file: 
        vectorizer = pickle.load(vectorizer_file)
        
    tflite_model_path = 'model_prep/quantized_model1.tflite'
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    return vectorizer, interpreter


def main_tweet():
    st.title("Disaster Detection with NLP")

    # Load model and text vectorization
    vectorizer, interpreter = vectorizer_and_load_model()

    # Display form for text input
    tweet_example = "Pada 1 Januari 2024, Jepun sekali lagi dilanda gempa bumi. Lebih 27 gempa berlaku dalam masa 2 jam"
    user_input = st.text_area("Enter text:", f'{tweet_example}')
    
    # Detect language
    lang_code = detect_language(user_input)[0]
    language_text = detect_language(user_input)[1]
    st.write(f"Language ID: \"{lang_code}\" ({language_text})")

    # Translate to English if not in English (en)
    if lang_code != 'en':
        with st.spinner("Translating to English..."):
            user_input = translate_to_english(user_input)
        # st.success("Translation complete!")
        st.write(f"Translation result: {user_input}")
    else:
        user_input = user_input
    
    
    st.markdown(
    """
    <style>
        div.stButton > button:first-child {
            background-color: purple;
            color: white;
            font-size: 10px;
            height: 3em;
            width: 10em;
            border-radius: 10px 10px 10px 10px;
        }
        div.stButton > button:first-child:hover, 
        div.stButton > button:first-child:active, 
        div.stButton > button:first-child:focus {
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)
    # Predict if there is input
    if st.button("Predict",):
        clean_text = preprocessing_text(user_input)
        
        # Transform input using CountVectorizer
        user_input_vec = vectorizer.transform([clean_text])

        # Prepare input for TensorFlow Lite model
        input_tensor_index = interpreter.get_input_details()[0]['index']
        interpreter.set_tensor(input_tensor_index, np.array(user_input_vec.toarray(), dtype=np.float32))

        # Perform inference
        interpreter.invoke()

        # Get output from TensorFlow Lite model
        output_tensor_index = interpreter.get_output_details()[0]['index']
        prediction = interpreter.get_tensor(output_tensor_index)[0][0]
        
        prediction_label = "Disaster" if prediction >= 0.5 else "Non-Disaster"
        
        st.success(f"Prediction Result: {prediction_label} (Probability: {prediction:.2f})", icon='✅')
   
    perform_ner(user_input)
