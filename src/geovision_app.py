import streamlit as st
from PIL import Image


def main():
    image_name = st.text_input('Enter path to file:')

    image = Image.open(image_name)
    st.image(image)

if __name__ == "__main__":
    main()