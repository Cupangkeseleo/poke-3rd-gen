import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

st.set_page_config(page_title='3rd Generation Pokemon Identifier', page_icon='gear', layout='wide', initial_sidebar_state='expanded')
st.markdown("<h1 style='text-align: center;'>3rd Generation Pokemon Identifier</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Upload an image of a 3rd generation Pokemon to identify it!</h5>", unsafe_allow_html=True)
st.image('Trio.png', width='stretch')
model_path = 'gen3_model_v2.pth'
data_dir = 'gen3_dataset'

class_names = ['Absol Pokemon', 'Aggron Pokemon', 'Altaria Pokemon', 'Anorith Pokemon', 'Armaldo Pokemon', 'Aron Pokemon', 'Azurill Pokemon', 'Bagon Pokemon', 'Baltoy Pokemon', 'Banette Pokemon', 'Barboach Pokemon', 'Beautifly Pokemon', 'Beldum Pokemon', 'Blaziken Pokemon', 'Breloom Pokemon', 'Cacnea Pokemon', 'Cacturne Pokemon', 'Camerupt Pokemon', 'Carvanha Pokemon', 'Cascoon Pokemon', 'Castform Pokemon', 'Chimecho Pokemon', 'Clamperl Pokemon', 'Claydol Pokemon', 'Combusken Pokemon', 'Corphish Pokemon', 'Cradily Pokemon', 'Crawdaunt Pokemon', 'Delcatty Pokemon', 'Deoxys Pokemon', 'Dusclops Pokemon', 'Duskull Pokemon', 'Dustox Pokemon', 'Electrike Pokemon', 'Exploud Pokemon', 'Feebas Pokemon', 'Flygon Pokemon', 'Gardevoir Pokemon', 'Glalie Pokemon', 'Gorebyss Pokemon', 'Groudon Pokemon', 'Grovyle Pokemon', 'Grumpig Pokemon', 'Gulpin Pokemon', 'Hariyama Pokemon', 'Huntail Pokemon', 'Illumise Pokemon', 'Jirachi Pokemon', 'Kecleon Pokemon', 'Kirlia Pokemon', 'Kyogre Pokemon', 'Lairon Pokemon', 'Latias Pokemon', 'Latios Pokemon', 'Lileep Pokemon', 'Linoone Pokemon', 'Lombre Pokemon', 'Lotad Pokemon', 'Loudred Pokemon', 'Ludicolo Pokemon', 'Lunatone Pokemon', 'Luvdisc Pokemon', 'Makuhita Pokemon', 'Manectric Pokemon', 'Marshtomp Pokemon', 'Masquerain Pokemon', 'Mawile Pokemon', 'Medicham Pokemon', 'Meditite Pokemon', 'Metagross Pokemon', 'Metang Pokemon', 'Mightyena Pokemon', 'Milotic Pokemon', 'Minun Pokemon', 'Mudkip Pokemon', 'Nincada Pokemon', 'Ninjask Pokemon', 'Nosepass Pokemon', 'Numel Pokemon', 'Nuzleaf Pokemon', 'Pelipper Pokemon', 'Plusle Pokemon', 'Poochyena Pokemon', 'Ralts Pokemon', 'Rayquaza Pokemon', 'Regice Pokemon', 'Regirock Pokemon', 'Registeel Pokemon', 'Relicanth Pokemon', 'Roselia Pokemon', 'Sableye Pokemon', 'Salamence Pokemon', 'Sceptile Pokemon', 'Sealeo Pokemon', 'Seedot Pokemon', 'Seviper Pokemon', 'Sharpedo Pokemon', 'Shedinja Pokemon', 'Shelgon Pokemon', 'Shiftry Pokemon', 'Shroomish Pokemon', 'Shuppet Pokemon', 'Silcoon Pokemon', 'Skitty Pokemon', 'Slaking Pokemon', 'Slakoth Pokemon', 'Snorunt Pokemon', 'Solrock Pokemon', 'Spheal Pokemon', 'Spinda Pokemon', 'Spoink Pokemon', 'Surskit Pokemon', 'Swablu Pokemon', 'Swalot Pokemon', 'Swampert Pokemon', 'Swellow Pokemon', 'Taillow Pokemon', 'Torchic Pokemon', 'Torkoal Pokemon', 'Trapinch Pokemon', 'Treecko Pokemon', 'Tropius Pokemon', 'Vibrava Pokemon', 'Vigoroth Pokemon', 'Volbeat Pokemon', 'Wailmer Pokemon', 'Wailord Pokemon', 'Walrein Pokemon', 'Whiscash Pokemon', 'Whismur Pokemon', 'Wingull Pokemon', 'Wurmple Pokemon', 'Wynaut Pokemon', 'Zangoose Pokemon', 'Zigzagoon Pokemon']

new_class_names = [name.replace(' Pokemon', '') for name in class_names]

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(new_class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

model, device = load_model()

def predict(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img_processed = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_processed)
        prob = torch.nn.functional.softmax(outputs, dim=1)
        top3_prob, top3_idx = torch.topk(prob, 3)

    return top3_prob, top3_idx

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h6 style='text-align: center;'>Choose an image...</h5>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Choose an image...", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed" 
    )

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', width='stretch')

    with st.spinner('Classifying...'):
        top3_prob, top3_idx = predict(image)

    st.success('Classification Complete!')

    winner_idx = top3_idx[0][0].item()
    winner_name = new_class_names[winner_idx]
    winner_score = top3_prob[0][0].item()
    
    st.header(f"It's... **{winner_name}**! ({winner_score*100:.1f}%)")
    
    st.write("### Top 3 Guesses:")
    for i in range(3):
        idx = top3_idx[0][i].item()
        prob = top3_prob[0][i].item()
        name = new_class_names[idx]
        col1, col2 = st.columns([1, 3])
        with col1:
            st.write(f"**{name}**")
        with col2:
            st.progress(int(prob * 100))