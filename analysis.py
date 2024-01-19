import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go


import prep_app.preprocessing as pre
from prep_app.translator import detect_language, translate_to_english
from prep_app.ner import perform_ner_gpe



# Define your preprocessing_text function
def preprocessing_text(text, use_stemmer=True):
    if detect_language(text) != 'en':
        text = translate_to_english(text)
    # Perform text cleaning
    cleaned_input_text = pre.clean_text(text, use_stemmer=use_stemmer)
    return cleaned_input_text

# Define your vectorizer_and_load_model function
def vectorizer_and_load_model():
    with open("model_prep/count_vectorizer.pkl", "rb") as vectorizer_file: 
        vectorizer = pickle.load(vectorizer_file)
        
    tflite_model_path = 'model_prep/quantized_model1.tflite'
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    return vectorizer, interpreter

# Function to predict text
def predict_text(text, vectorizer, interpreter):
    clean_text = preprocessing_text(text) 
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
    
    return "Disaster" if prediction >= 0.5 else "Non-Disaster"


def visualize_ner_entities(df, prediction_column, labels=None):
    # Filter rows based on 'prediction_column' and 'labels'
    disaster_rows = df[df[prediction_column] == labels]

    # Flatten list of entities
    all_entities = [entity for entities in disaster_rows['ner_text'] for entity in entities]

    # count frequency of entity
    entity_counts = {}
    for entity in all_entities:
        entity_counts[entity] = entity_counts.get(entity, 0) + 1

    #change dictionary to dataframe
    entity_counts_df = pd.DataFrame(list(entity_counts.items()), columns=['Entity', 'Count'])
    entity_counts_df = entity_counts_df.sort_values(by='Count', ascending=False)
    # st.dataframe(entity_counts_df)
    
    fig = px.bar(entity_counts_df, x='Count', y='Entity',
                 labels={'Entity': 'Location', 'Count': 'Count'},
                 title='Number of Entities in Disaster Predictions',
                 color_discrete_sequence=['#bb2b77'],
                 orientation='h')
    fig.update_layout(yaxis={'categoryorder':'total ascending'},
                      xaxis={'tickmode': 'linear', 'dtick': 1})
    st.plotly_chart(fig)


def visualize_piechart(labels=None, values= None, 
                       explode=(0.05, 0), colors=['#bb2b77','#95cbee'],
                       var=None):
    
    ## Create a Pie chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, 
                                 pull=explode, 
                                 marker=dict(
                                     colors=colors,
                                     line=dict(color='#000000', width=2)))])

    # Adjusting the position of the horizontal legend
    fig.update_layout(
        legend=dict(
            x=0.5, y=1.15, 
            xanchor='center',
            orientation='h',  # Horizontal orientation
            bgcolor='rgba(211,211,211,0.3)',  # legend background
            font=dict(size=12)    
        ),
        title=dict(text=f'{var} Distribution', x=0.5, y=0.95) 
    )
    
    st.plotly_chart(fig)

@st.cache_data
def load_data_with_predictions(uploaded_file=None, column_to_predict=None):
    #read the file
    df = pd.read_csv(uploaded_file)
    
    # Perform predictions and add a new column
    vectorizer, interpreter = vectorizer_and_load_model()
    df['Prediction'] = df[column_to_predict].apply(lambda x: predict_text(x, vectorizer, interpreter))
    list_translated_text = df[column_to_predict].apply(lambda x: preprocessing_text(x, use_stemmer=False))
    df['ner_text'] = list_translated_text.apply(lambda x: perform_ner_gpe(x))
    return df

    
# Streamlit app
def main():
    st.title("Disaster Prediction App")

    row1 = st.columns(2)
    row2 = st.columns(2)
    with row1[0]:
        # File upload
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "csv"])
        # User input for column name
        column_to_predict = st.text_input("Enter the column name to predict", "text")
        

    # Check if 'clicked' is not present in the session state
    if 'clicked' not in st.session_state:
        # If not present, initialize it to True
        st.session_state.clicked = True
    def click_button():
         # Set 'clicked' to True when the button is clicked
        st.session_state.clicked = True
    
    with row1[0]:
        st.button("Submit", on_click=click_button)
    
    if st.session_state.clicked:
        if uploaded_file is not None and column_to_predict:
            # Read the file
            with row1[1]:
                st.write("DataFrame with Predictions")
                df=load_data_with_predictions(uploaded_file, column_to_predict)
                start_row, end_row=st.slider('Number of rows to display', 0, df.shape[0], (0, 100))
                st.dataframe(df.iloc[start_row:end_row])

                # Download button for the new dataframe
                st.download_button(
                    label="Download Predictions as CSV",
                    data=df.to_csv().encode(),
                    file_name="predictions.csv",
                    key="download_predictions"
                )
            
            with row2[0]:
                labels = df['Prediction'].value_counts().index
                values= df['Prediction'].value_counts().values
                visualize_piechart(labels, values,  var='Prediction Tweets')
            with row2[1]:
                visualize_ner_entities(df, prediction_column='Prediction', labels='Disaster')
        
        else:
            st.warning("Please choose a file and enter a valid column name.")


    if __name__ == "__main__":
        main()
