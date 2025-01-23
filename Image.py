import streamlit as st
from sklearn.cluster import KMeans
import requests
from PIL import Image
import numpy as np

st.title("Image Segmentation with KMeans")

col1, col2 = st.columns(2)



with col1:
    
    img_link = st.text_input("Image URL (Press Enter to apply)")
    if img_link != "":
        img_pil = Image.open(requests.get(img_link, stream=True).raw)
        st.image(img_link, caption = "Origin al Image")
    
    
with col2:
    if img_link != "":
        img = np.array(img_pil)
        X = img.reshape(-1,4)
        K = st.slider(label = "K", min_value = 2, max_value = 10)
        kmeans_model = KMeans(n_clusters = K, n_init = "auto")
        kmeans_model.fit(X)
        img_new = kmeans_model.cluster_centers_[kmeans_model.labels_].reshape(img.shape[0], img.shape[1], img.shape[2]).astype(np.uint8)
        img_new = Image.fromarray(img_new)
        st.image(img_new, caption = "Segmented Image")

    
