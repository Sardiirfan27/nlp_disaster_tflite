import streamlit as st
from spacy import displacy
import spacy
from translator import detect_language, translate_to_english

nlp = spacy.load("en_core_web_sm")

def get_html(html):
    return f'<div style="max-width: 1000px; margin: auto; overflow-x: auto;">{html}</div>'

def perform_ner(text):

    st.title("Named Entity Recognition")
    
    doc = nlp(text)
    labels = [ent.label_ for ent in doc.ents]
    exp = st.expander("Select entity labels")
    label_select = exp.multiselect(
        "Entity labels",
        options=list(set(labels)),
        default=list(set(labels)),
        key="ner_label_select",
    )
    displacy_options = {"ents": label_select}
    html = displacy.render(
        doc,
        style="ent",
        options=displacy_options,
    )
    st.write(get_html(html), unsafe_allow_html=True)



if __name__ == "__main__":
    #testing
    st.title("Named Entity Recognition with Streamlit")
    
    # Input teks dari pengguna
    text_input = st.text_area("Masukkan teks:", "Gempa bumi di Indonesia.")
    
    lang_code = detect_language(text_input)
    st.write(f"Bahasa terditeksi: {lang_code}")

    if lang_code != 'en':
        with st.spinner("Melakukan terjemahan ke bahasa Inggris..."):
            text_input = translate_to_english(text_input)
        st.success("Terjemahan selesai!")
        st.write(f"Hasil terjemahan: {text_input}")
    
    else:
        text_input

    # Tombol "Submit"
    if st.button("Submit"):
        perform_ner(text= text_input)