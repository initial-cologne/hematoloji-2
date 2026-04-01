import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# --- Sayfa Yapılandırması ---
st.set_page_config(page_title="Hematoloji Analiz", page_icon="🔬")
st.title("Hematolojik Analiz Arayüzü")

# 1. Model Hazırlığı (Daha önce yaptığımız gibi)
siniflar = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']


@st.cache_resource  # Modelin her seferinde yeniden yüklenmesini engeller, hızı artırır
def model_yukle():
    m = models.resnet18()
    m.fc = torch.nn.Linear(m.fc.in_features, len(siniflar))
    m.load_state_dict(torch.load("dinamik_resnet_modeli.pth", map_location='cpu'))
    m.eval()
    return m


model = model_yukle()

# 2. Görüntü Dönüşümü
donusum = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Giriş Yöntemi Seçimi
sekme1, sekme2 = st.tabs(["📁 Dosya Yükle", "🌐 URL ile Analiz"])
goruntu = None

with sekme1:
    yuklenen_dosya = st.file_uploader("Bir hücre fotoğrafı seçin...", type=["jpg", "jpeg", "png"])
    if yuklenen_dosya:
        goruntu = Image.open(yuklenen_dosya).convert('RGB')

with sekme2:
    url_adresi = st.text_input("Görüntü URL'sini buraya yapıştırın:")
    if url_adresi:
        try:
            yanit = requests.get(url_adresi)
            goruntu = Image.open(BytesIO(yanit.content)).convert('RGB')
        except Exception as e:
            st.error(f"Görüntü indirilemedi. Lütfen URL'yi kontrol edin. Hata: {e}")

# 4. Analiz ve Tahmin Süreci
if goruntu:
    st.image(goruntu, caption='Analiz Edilen Hücre', use_container_width=True)

    with st.spinner('Yapay zeka morfolojiyi inceliyor...'):
        input_tensor = donusum(goruntu).unsqueeze(0)
        with torch.no_grad():
            ciktilar = model(input_tensor)
            olasiliklar = torch.nn.functional.softmax(ciktilar[0], dim=0)
            tahmin_indisi = torch.argmax(olasiliklar).item()

    # Sonuçların Gösterilmesi
    st.success(f"Teşhis: **{siniflar[tahmin_indisi]}**")
    st.progress(float(olasiliklar[tahmin_indisi]))  # Görsel bir bar ekler
    st.write(f"Güven Skoru: %{olasiliklar[tahmin_indisi] * 100:.2f}")