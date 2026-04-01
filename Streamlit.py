import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# --- Sayfa Yapılandırması ---
st.set_page_config(page_title="Hematoloji Analiz", page_icon="🔬")
st.title("Hematolojik Analiz Arayüzü")
st.write("Lökosit grubu hücrelere ait görüntüyü bilgisayarınızdan yükleyebilir veya bir web URL'si girebilirsiniz.")

# --- Model Hazırlığı ---
siniflar = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']


@st.cache_resource
def model_yukle():
    m = models.resnet18()
    m.fc = torch.nn.Linear(m.fc.in_features, len(siniflar))
    m.load_state_dict(torch.load("dinamik_resnet_modeli.pth", map_location='cpu'))
    m.eval()
    return m


model = model_yukle()

donusum = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Sekme (Tab) Yapısı ---
sekme1, sekme2 = st.tabs(["📁 Dosya Yükle", "🌐 URL ile Analiz"])
goruntu = None

with sekme1:
    yuklenen_dosya = st.file_uploader("Bir hücre fotoğrafı seçin...", type=["jpg", "jpeg", "png"])
    if yuklenen_dosya:
        goruntu = Image.open(yuklenen_dosya).convert('RGB')

with sekme2:
    url_adresi = st.text_input("Görüntü URL'sini buraya yapıştırın (Örn: https://.../resim.jpg):")
    if url_adresi:
        try:
            yanit = requests.get(url_adresi)
            goruntu = Image.open(BytesIO(yanit.content)).convert('RGB')
        except Exception as e:
            st.error("Görüntü indirilemedi. Lütfen URL'nin doğrudan bir görsele ait olduğundan emin olun.")

# --- Analiz ve 3'lü Tahmin ---
if goruntu is not None:
    st.image(goruntu, caption='Analiz Edilen Hücre', use_container_width=True)

    with st.spinner('morfoloji inceleniyor...'):
        input_tensor = donusum(goruntu).unsqueeze(0)
        with torch.no_grad():
            ciktilar = model(input_tensor)
            olasiliklar = torch.nn.functional.softmax(ciktilar[0], dim=0)

            # En yüksek 3 tahmini ve indekslerini alıyoruz
            top3_olasilik, top3_indis = torch.topk(olasiliklar, 3)

    st.subheader("🔬 Ayırıcı Tanı Sonuçları")

    # 3 tahmini döngü ile ekrana yazdırıyoruz
    for i in range(3):
        indis = top3_indis[i].item()
        skor = top3_olasilik[i].item()
        sinif_adi = siniflar[indis]

        # En yüksek ihtimali daha vurgulu (yeşil), diğerlerini bilgi (mavi) tonunda gösteriyoruz
        if i == 0:
            st.success(f"**1. {sinif_adi}** - Güven Skoru: %{skor * 100:.2f}")
            st.progress(float(skor))
        else:
            st.info(f"**{i + 1}. {sinif_adi}** - Güven Skoru: %{skor * 100:.2f}")
            st.progress(float(skor))