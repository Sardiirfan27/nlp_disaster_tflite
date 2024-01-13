from wordcloud import STOPWORDS
from nltk.stem import PorterStemmer
import re

def remove_stopword(text, words_to_remove=None):
    # Add words_to_remove to STOPWORDS
    if words_to_remove:
        STOPWORDS.update(words_to_remove)
        
    # Tokenize the text by splitting it. Remove empty strings and stopwords
    tokens = [token for token in re.split(r'\s+', text) if token != '' if token not in STOPWORDS]
    
    # Join tokens back into text
    processed_text = ' '.join(tokens)
    
    return processed_text

def apply_stemming(text):
    """
    This function applies stemming to the input text using Porter Stemmer.

    Parameters:
    - text: Input text

    Returns:
    - stemmed_text: Text after stemming
    """
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in re.split(r'\s+', text) if token != '']
    stemmed_text = ' '.join(tokens)
    return stemmed_text

def clean_text(text, words_to_remove=None, use_stemmer=False):
    """
    This function cleans text by removing special characters, converting text to lowercase,
    and performing additional cleaning steps as needed.

    Parameters:
    - text: Text to be cleaned
    - words_to_remove: List of words to be removed, e.g., ['will', 'one']
    - use_stemmer: Boolean, whether to use stemming or not

    Returns:
    - cleaned_text: The text after being cleaned
    """

    text = text.lower() # Convert text to lowercase
    text = remove_stopword(text,words_to_remove)#remove stopword
    text = re.sub(r'&\w+;', ' ', text) # Remove HTML entities like &amp;
    text = re.sub(r'@\S+', '', text) # Remove mentions (Twitter handles) starting with @
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text) # Remove URLs
    text = re.sub(r'\d', '', text) # Remove digits (numeric characters)
    
    # Remove non-alphanumeric characters (except spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Apply stemming if specified
    if use_stemmer:
        text= apply_stemming(text)
    
    # Replace multiple spaces with a single space and strip leading/trailing spaces
    cleaned_text = re.sub(r'\s+', ' ', text).strip()

    return cleaned_text


def clean_text_df(df, col_name, words_to_remove=None, use_stemmer=False):
    """
    This function cleans the text column in a dataframe using the clean_text function.

    Parameters:
    - df: DataFrame to be cleaned
    - col_name: Name of the text column in the dataframe
    - words_to_remove: List of words to be removed, e.g., ['will', 'one', 'body']
    - use_stemmer: Boolean, whether to use stemming or not

    Returns:
    - DataFrame with the cleaned text column
    """
    df_cleaned = df.copy()
    df_cleaned[col_name] = df_cleaned[col_name].apply(lambda x: clean_text(x, words_to_remove, use_stemmer))
    return df_cleaned
