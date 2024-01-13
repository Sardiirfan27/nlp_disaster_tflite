from googletrans import Translator

def detect_language(text):
    try:
        translator_detect = Translator().detect(text)
        lang_detection = translator_detect.lang
        return lang_detection
    except TypeError:
        # Tangani TypeError dan beri nilai default
        return 'en'  # Nilai default jika terjadi kesalahan

def translate_to_english(text):
    translator = Translator()
    translation = translator.translate(text, dest='en')
    return translation.text