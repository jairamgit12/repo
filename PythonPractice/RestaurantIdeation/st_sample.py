import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# #..SAMPLE1..
# # st.title("Streamlit Sample Project")

# #..SAMPLE2..
# # st.title("Textbox Example")

# # text_input = st.text_input("Enter some text: ")
# # st.write("Text entered:", text_input)

# #..SAMPLE3..
# # st.title("Camera Input Example")

# # st.camera_input("Take a Picture")

# #  ..SAMPLE3.. draw a slider component
# st.title("Slider Example")

# num_points = st.slider("Number of points", min_value=100, max_value=1000, value=500, step=100)

st.title("Line Plot Example")
# Load sample data
data = pd.DataFrame({
 "x": [1,1.5,2,2.5,3,3.5,4], 
 "y": [10, 15,20,25,30,35,40]})

# Plot the data
fig, ax = plt.subplots()
ax.scatter(data["x"], data["y"])
# Show the plot in the Streamlit app
st.pyplot(fig)
