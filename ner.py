import streamlit as st
from spacy import displacy
import spacy
from translator import detect_language, translate_to_english



def get_html(html):
    return f'<div style="max-width: 1000px; margin: auto; overflow-x: auto;">{html}</div>'

@st.cache_data
def processing_text(spacy_model,text):
    nlp=spacy.load(spacy_model)
    doc=nlp(text)
    return doc

def visualize_ner(doc, labels= tuple(), 
                  title = "Named Entity Recognition",
                  displacy_options: dict = {},
                  ):
    
    if title:
        st.header(title)

    if not labels:
        st.warning("The parameter 'labels' should not be empty or None.")
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


def perform_ner(text):
    
    doc= processing_text("en_core_web_sm",text)
    labels = [ent.label_ for ent in doc.ents]
    visualize_ner(doc, labels=labels)



if __name__ == "__main__":
    #testing
    st.title("Named Entity Recognition with Streamlit")
    
    # Input teks dari pengguna
    text_input = st.text_area("Masukkan teks:", "Google was founded in September 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a California privately held company on September 4, 1998, in California. Google was then reincorporated in Delaware on October 22, 2002.")
    
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
    perform_ner(text= text_input)