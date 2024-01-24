import streamlit as st
from spacy import displacy
import spacy
from prep_app.translator import detect_language, translate_to_english

# Function to wrap HTML in a div for styling
def get_html(html):
    return f'<div style="max-width: 1000px; margin: auto; overflow-x: auto;">{html}</div>'

# Function for processing text using SpaCy model
@st.cache_data
def processing_text(spacy_model, text):
    nlp = spacy.load(spacy_model)
    doc = nlp(text)
    return doc

# Function to visualize Named Entity Recognition (NER) results
def visualize_ner(doc, labels= tuple(), 
                  title="Named Entity Recognition",
                  displacy_options: dict = {}):
    
    if title:
        st.header(title)

    if not labels:
        st.warning("The parameter 'labels' is empty")
    else:
        exp = st.expander("Select entity labels")
        label_select = exp.multiselect(
            "Entity labels",
            options=labels,
            default=list(labels),
        )

        displacy_options["ents"] = label_select
        html = displacy.render(
            doc,
            style="ent",
            options=displacy_options,
        )
        style = "<style>mark.entity { display: inline-block }</style>"
        st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)

# Function to perform NER on the given text
def perform_ner(text):
    doc = processing_text("en_core_web_sm", text)
    labels = [ent.label_ for ent in doc.ents]
    return visualize_ner(doc, labels=labels)
    
def perform_ner_gpe(text):
    #perform NER 
    labels = ["GPE"]
    doc = processing_text("en_core_web_sm", text)
    #entity labels
    entities = [ent.text for ent in doc.ents if ent.label_ in labels]
    return entities

# Main Streamlit app
if __name__ == "__main__":
    st.title("Named Entity Recognition")
    
    # User input for text
    text_input = st.text_area("Enter text:", "Google was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a California privately held company on September 4, 1998, in California. Google was then reincorporated in Delaware on October 22, 2002.")
    
    # Detect language of input text
    lang_code = detect_language(text_input)
    st.write(f"Detected language: {lang_code}")

    # Translate to English if the language is not English
    if lang_code != 'en':
        with st.spinner("Translating to English..."):
            text_input = translate_to_english(text_input)
        st.success("Translation complete!")
        st.write(f"Translation result: {text_input}")
    else:
        text_input

    # Button to trigger Named Entity Recognition (NER)
    perform_ner(text=text_input)
