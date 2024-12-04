import streamlit as st
from utils import load_model,load_image
from remove_bg import remove_bg
from io import BytesIO

st.set_page_config('Background Remover',layout='centered')
MAX_SIZE = 1024*1024
extensions = ['png','jpg','jpeg','webp']
model_options = ['Anime','General','Potrait']

st.title('Background Remover')
st.write('This project implement deep learning model (DeepLabV3 MobileNetV3 from [PyTorch](https://pytorch.org/vision/stable/models/generated/torchvision.models.segmentation.deeplabv3_mobilenet_v3_large.html#torchvision.models.segmentation.deeplabv3_mobilenet_v3_large)) for removing image background, there are 3 mode options:')
st.markdown("""
- Anime: For anime figures (Fine-tuned model using this [Anime Image Dataset](https://huggingface.co/datasets/skytnt/anime-segmentation))
- General: For natural, general image (No Fine-Tuning)
- Potrait: For human potrait image (Fine-tuned model using this [Kaggle Dataset](https://www.kaggle.com/datasets/kapitanov/easyportrait))
""")

st.markdown("""
<small>Note: The model is still experimental and might be slow.</small>
""",unsafe_allow_html=True)

selected_model = st.selectbox('Please select mode',model_options)

model = load_model(selected_model)


uploaded = st.file_uploader('Upload an image here',
                            extensions,
                            accept_multiple_files=False)

if uploaded:
    if uploaded.size > MAX_SIZE:
        st.error('Uploaded image size is too large, please upload an smaller image')
    else:
        filename:str = uploaded.name
        raw = load_image(uploaded)
        cleaned = remove_bg(model,raw)
        col1,col2 = st.columns([1,1],vertical_alignment='center')
        with col1:
            st.image(raw,'Before')
        with col2:
            st.image(cleaned,'After')
        byte = BytesIO()
        cleaned.save(byte,format='PNG')
        byte.seek(0)
        for ext in extensions:
            if filename.endswith(ext):
                filename = filename.replace(ext,'')
        col1,col2,col3 = st.columns([1,1,1],vertical_alignment='center')
        with col2:
            st.download_button(
                'Download the Cleaned Image',
                byte,
                file_name=f'{filename}_cleaned.png',
                mime='image/png'
            )
    