import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

model_fraud = pickle.load(open('model_deteksi_komentar.sav','rb'))

tfidf = TfidfVectorizer

loaded_vec = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("new_selected_feature_tf-idf.sav", "rb"))))
st.set_page_config(
    page_title="Deteksi Komentar Hate Speech Pada Instagram",
    page_icon=":star:",
    
)
import requests
from io import BytesIO

from PIL import Image
instagram_logo_url = 'https://kebudayaan.kemdikbud.go.id/bpcbbali/wp-content/uploads/sites/26/2020/07/logo-ig-instagram-png-transparent-instagram-images-pluspng-3.png'
response = requests.get(instagram_logo_url)
if response.status_code == 200:
    instagram_logo = Image.open(BytesIO(response.content))

    # Resize the image to a smaller size
    smaller_logo = instagram_logo.resize((100, 100))  # Adjust the size as needed

    # Display the resized logo
    st.image(smaller_logo)
else:
    st.write("Gagal mengunduh logo Instagram")

def main():
    st.title("Deteksi Komentar Hate Speech Pada Instagram")
    message = st.text_area("Masukan Komentar")

    if st.button("Deteksi Komentar"):
        predict_fraud = model_fraud.predict(loaded_vec.fit_transform([message]))
    
        if predict_fraud == 0:
            fraud_detection = 'Komentar terindikasi hatespeech'
        elif predict_fraud == 1:
            fraud_detection = 'Komentar tidak terindikasi hatespeech'
        else:
            fraud_detection = 'inputanmu salah'
        st.success(fraud_detection)

    # Menampilkan teks paling bawah
    st.text("Project by Amalia Rachmadana Ismail Statistika UII")
    
if __name__ == "__main__":
    main()
