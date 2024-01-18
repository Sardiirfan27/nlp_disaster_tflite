from googletrans import Translator, LANGUAGES

def detect_language(text):
    try:
        translator_detect = Translator().detect(text)
        lang_id = translator_detect.lang        
        detect_language = LANGUAGES[lang_id]
        return lang_id, detect_language

    except TypeError:
        # Handle TypeError and provide a default value
        return 'unknown'  # Default value in case of an error

def translate_to_english(text):
    translator = Translator()
    translation = translator.translate(text, dest='en')
    return translation.text
