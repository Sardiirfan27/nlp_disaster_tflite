# home.py
import streamlit as st
import streamlit.components.v1 as components
# from PIL import Image   

def home_page():
    centered_title = """
    <h1 style="text-align: center;">Disaster Analysis and Prediction from Tweets</h1>
    """
    st.markdown(centered_title, unsafe_allow_html=True)
    
    components.html("""
        <style>
        .carousel-item {
            position: relative;
        }

        .overlay {
            position: absolute;
            top: 50%;
            left: 0;
            width: 100%;
            transform: translateY(-50%);
            background: rgba(0, 0, 0, 0.5); /* Semi-transparent black background */
            color: white;
            text-align: center;
            padding: 20px;
        }
        </style>
       <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
       <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"></script>
        <!-- Carousel -->
        <div id="demo" class="carousel slide" data-bs-ride="carousel">

        <!-- Indicators/dots -->
        <div class="carousel-indicators">
            <button type="button" data-bs-target="#demo" data-bs-slide-to="0" class="active"></button>
            <button type="button" data-bs-target="#demo" data-bs-slide-to="1"></button>
            <button type="button" data-bs-target="#demo" data-bs-slide-to="2"></button>
        </div>
        
        <!-- The slideshow/carousel -->
        <div class="carousel-inner">
            <div class="carousel-item active"> 
                <img src="https://insideclimatenews.org/wp-content/uploads/2023/03/wildfire_thibaud-mortiz-afp-getty-scaled.jpg" class="d-block w-100" alt="Slide 1">
                <div class="overlay">
                <h3>Wildfire</h3>
                <p>A forest fire in Louchats, southwestern France, on July 17, 2022.</p>
                Credit: Thibaud Moritz/AFP via Getty Images
                </div>
            </div>
            <div class="carousel-item">
            <img src="https://asset.kompas.com/crops/cKW0Rlts9q27vXhl_gho26h6bm0=/0x0:1992x1328/750x500/data/photo/2020/03/11/5e689e40e618e.jpg" class="d-block w-100" alt="Slide 2">
            <div class="overlay">
                <h3>Tsunami</h3>
                <p>Tsunami waves hit Miyako City in Iwate Prefecture after a 9.0 magnitude earthquake rocked the Tohoku region, March 11 2011. (HO NEW/REUTERS)
                </p>
            </div> 
            </div>
            <div class="carousel-item">
                <img src="https://cdn.britannica.com/34/127134-050-49EC55CD/Building-foundation-earthquake-Japan-Kobe-January-1995.jpg" class="d-block w-100" alt="Slide 3">
            <div class="overlay">
                <h3>earthquake</h3>
                <p>Kōbe earthquake of 1995, (Jan. 17, 1995) large-scale earthquake in the Ōsaka-Kōbe (Hanshin) metropolitan area of western Japan. 
                    (www.britannica.com)
                </p>
                
            </div>  
            </div>
        </div>
        
        <!-- Left and right controls/icons -->
        <button class="carousel-control-prev" type="button" data-bs-target="#demo" data-bs-slide="prev">
            <span class="carousel-control-prev-icon"></span>
        </button>
        <button class="carousel-control-next" type="button" data-bs-target="#demo" data-bs-slide="next">
            <span class="carousel-control-next-icon"></span>
        </button>
        </div>
    """,height=400
    )
    #opening the image
    # image = Image.open('disaster.png')
    # st.image(image,caption="disaster", use_column_width=True)
   

    paragraph = """
    <p>In the current digital era, social media platforms have evolved into dynamic spaces where individuals share thoughts, experiences, and, most importantly, real-time updates during emergencies and disasters. Among these platforms, Twitter stands out as a valuable source of data that, if effectively harnessed, can significantly impact emergency response, disaster management, humanitarian aid organizations and news media. 
    Therefore, in this project, we are attempting to predict disasters from tweets.
    </p>
    <p>The project's overarching objectives include developing a reliable classification model that significantly contributes to the efficiency of disaster response and management.
    Furthermore, in this project, we can also perform geographic analysis using Named Entity Recognition (NER) to identify areas affected by disasters. This facilitates efficient and targeted emergency response efforts, aiding humanitarian organizations in identifying disaster-affected regions. 
    The key points that encapsulate the project's goals are as follows:
    </p>
    """
    st.markdown(paragraph, unsafe_allow_html=True)

    markdown_text = """
    - **Enhancing Model Robustness:** Implementing techniques to enhance the robustness of the classification model, ensuring consistent performance across various types of disaster-related tweets and linguistic variations.
    - **Multilingual Support:** Exploring and integrating multilingual capabilities to ensure the model's effectiveness in analyzing tweets written in different languages, broadening its applicability to diverse global contexts.
    - **Geographical Analysis:** Enabling the model to analyze and identify disaster-prone regions, enhancing its capability to provide insights into the geographical areas affected by disasters.
    """
    st.markdown(markdown_text)


    # bootstrap 5 collapse 
    components.html(
        """
       <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/css/bootstrap.min.css" rel="stylesheet">
       <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.0.2/dist/js/bootstrap.bundle.min.js"></script>
       <div class="m-4">
        <div class="accordion" id="myAccordion">
            <div class="accordion-item">
                <h2 class="accordion-header" id="headingOne">
                    <button type="button" class="accordion-button collapsed" data-bs-toggle="collapse" data-bs-target="#collapseOne"><b>What is Disaster?</b></button>									
                </h2>
                <div id="collapseOne" class="accordion-collapse collapse" data-bs-parent="#myAccordion">
                    <div class="card-body">
                        <p>A disaster is a sudden, catastrophic event that causes significant disruption, destruction, and distress, often resulting in serious harm to people, property, and the environment. 
                        Disasters can take various forms, including natural disasters and human-made disasters
                        Fundamentally, disasters can be categorized into two main types: human-caused disasters and natural disasters. 
                        <a href="https://www.samhsa.gov/find-help/disaster-distress-helpline/disaster-types" target="_blank">Learn more.</a></p>
                        <strong>1. Human-Caused Disasters:</strong>
                        <ul>
                            <li>
                                <strong>Causes:</strong> Result from intentional or negligent human actions, such as industrial accidents, acts of terrorism, shootings, and incidents of mass violence.
                            </li>
                            <li>
                                <strong>Impacts:</strong> Lead to loss, trauma, and may prompt evacuations, overwhelming behavioral health resources in affected communities.
                            </li>
                        </ul>

                        <strong>2. Natural Disasters:</strong>
                        <ul>
                            <li>
                                <strong>Causes:</strong> Triggered by natural forces, including tornadoes, hurricanes, floods, wildfires, earthquakes, and droughts.
                            </li>
                            <li>
                                <strong>Impacts:</strong> Pose severe threats to life and property, often necessitating emergency measures and declarations.
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
            <div class="accordion-item">
                <h2 class="accordion-header" id="headingTwo">
                    <button type="button" class="accordion-button" data-bs-toggle="collapse" data-bs-target="#collapseTwo"><b>How to Predict Disaster Tweets?</b></button>
                </h2>
                <div id="collapseTwo" class="accordion-collapse collapse show" data-bs-parent="#myAccordion">
                    <div class="card-body">
                        <p>To predict tweets related to disaster, you can use the tweet and analysis menus.
                         <ul>
                            <li>
                                In the <b>tweet menu</b>, you can input text or tweets for prediction. 
                                Furthermore, if your input is in a language other than English, 
                                it will be translated into English before the prediction results appear.
                            </li>
                            <li>
                                In the <b>analysis menu</b>, you can upload a CSV file containing text or 
                                tweets with the column name "text." This will automatically predict each row 
                                and identify geographical entities from the text.
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
        </div>
        </div>

        """,
        height=600,
    )

if __name__ == "__main__":
    home_page()
