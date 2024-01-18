import streamlit as st
import hydralit_components as hc
import datetime
import numpy as np
import matplotlib.pyplot as plt

#make it look nice from the start
st.set_page_config(layout='wide',initial_sidebar_state='collapsed',)

# specify the primary menu definition
menu_data = [
    {'icon': "fa fa-folder-open", 'label':"Upload File"},
    {'id':'Copy','icon':"fa fa-twitter",'label':"Tweet"},
    {'icon': "fa-solid fa-radar",'label':"Dropdown1", 'submenu':[{'id':' subid11','icon': "fa fa-paperclip", 'label':"Sub-item 1"},{'id':'subid12','icon': "💀", 'label':"Sub-item 2"},{'id':'subid13','icon': "fa fa-database", 'label':"Sub-item 3"}]},
    {'icon': "far fa-chart-bar", 'label':"Chart"},#no tooltip message
    {'icon': "fas fa-tachometer-alt", 'label':"Dashboard",'ttip':"I'm the Dashboard tooltip!"}, #can add a tooltip message
    {'icon': "fa-solid fa-radar",'label':"Dropdown2", 'submenu':[{'label':"Sub-item 1", 'icon': "fa fa-meh"},{'label':"Sub-item 2"},{'icon':'🙉','label':"Sub-item 3",}]},
]

over_theme = {'txc_inactive': '#FFFFFF','menu_background':'purple'}
menu_id = hc.nav_bar(  
    menu_definition=menu_data,
    override_theme=over_theme,
    home_name='Home',
    # login_name='Logout',
    hide_streamlit_markers=True, #will show the st hamburger as well as the navbar now!
    sticky_nav=True, #at the top or not
    sticky_mode='pinned', #jumpy or not-jumpy, but sticky or pinned
)


# Check if the Home menu item is clicked
if menu_id == 'Home':
    st.markdown("# Welcome to the Home Page!")
    
if menu_id == 'Chart':
    st.markdown("# Welcome to the Home Page!")
    
    # Display a sample chart (you can replace this with your own chart code)
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plt.plot(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Sample Chart')
    
    # Display the chart in Streamlit
    st.pyplot(plt)
    
if st.button('click me'):
  st.info('You clicked at: {}'.format(datetime.datetime.now()))


if st.sidebar.button('click me too'):
  st.info('You clicked at: {}'.format(datetime.datetime.now()))

#get the id of the menu item clicked
st.info(f"{menu_id}")