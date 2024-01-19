from wordcloud import STOPWORDS
from nltk.stem import PorterStemmer
import re
import pandas as pd

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



def generate_ngrams(text, n_gram=1):
    # Tokenize the text, remove empty strings and stopwords
    token = [token for token in text.split(' ') if token != '' if token not in STOPWORDS]
    
    # Generate n-grams using a sliding window approach
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    
    # Combine the n-grams into a list of strings
    return [' '.join(ngram) for ngram in ngrams]


def ngrams_frequencies(text_data, n_grams=1, name='unigram'):
    """
    This function creates a DataFrame containing n-grams frequencies based on the provided text data.

    Parameters:
    - text_data: Iterable of text data
    - n_grams: Size of n-grams (default is 1 for unigrams)
    - name: Name to use for the n-grams column in the DataFrame (default is 'unigram')

    Returns:
    - DataFrame: DataFrame containing n-gram frequencies
    """
    # Initialize an empty dictionary to store word frequencies
    ngrams_dict = {}

    # Iterate over each text in the provided text data
    for text in text_data:
        # Generate n-grams for each word in the text
        for ngram in generate_ngrams(text, n_grams):
            # Update the frequency count for each n-gram in the word_dict dictionary
            ngrams_dict[ngram] = ngrams_dict.get(ngram, 0) + 1

    # Convert the n-gram dictionary to a DataFrame
    df_ngrams = pd.DataFrame(list(ngrams_dict.items()), columns=[name, f'{name}_counts'])

    return df_ngrams
