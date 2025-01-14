import streamlit as st
import subprocess
import configparser

config = configparser.ConfigParser()

# Streamlit page for user inputs
def user_input_page():
    st.title("Research Topic and Websites Input")

    # Input for research topic
    topic = st.text_input("Enter the research topic:")

    # Input for list of websites
    websites = st.text_area("Enter the list of websites (one per line):")
    websites = websites.splitlines()
    
    config['DEFAULT'] = {'DEFAULT_TOPIC': "\"{0}\"".format(topic),
                         'INITIAL_WEBSITES': websites}

    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    # Button to load and run web_search.py
    if st.button("Execute Web Research"):
        # Execute web_search.py and stream output
        process = subprocess.run(["python3", "web_search.py"], stderr=subprocess.PIPE, text=True)
        error_message = process.stderr

        # Stream the output in real-time
        # for line in process.stdout:
            # st.write(line)  # Display each line of output as it is produced
        
        # Wait for the process to complete
        # process.wait()
        
        # Check for any errors
        if process.returncode != 0:
            st.error(f"Error occurred: {error_message}")
        
        st.success("Web search executed successfully!")

# Call the user input page function
user_input_page()