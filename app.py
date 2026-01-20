import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from models.model import MobileNetV2_Alzheimer_Classfier
import torch.functional as F
from torch import nn

mean = [0.1650, 0.1650, 0.1650]
std = [0.1761, 0.1761, 0.1761]

classes_map = {0:'Mild Dementia',
               1:'Moderate Dementia',
               2:'Non Demented',
               3:'Very mild Dementia'}


# تحميل الموديل
model = MobileNetV2_Alzheimer_Classfier(4)
model.load_state_dict(torch.load("models/model_moblienet_alzheimer.pth", map_location=torch.device('cpu')))
model.eval()  # مهم: inference mode

# تحميل صورة من المستخدم
upload_image = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if upload_image is not None:
    # فتح الصورة وتحويلها إلى RGB
    image = Image.open(upload_image).convert("RGB")
    st.image(image, caption="Uploaded Image",width="stretch")

    # تحويل الصورة إلى Tensor بالشكل الصحيح
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # حسب ما الموديل تدرب عليه
        transforms.ToTensor(),           # HxWxC -> CxHxW + normalize [0,1]
         transforms.Normalize(mean=mean,std=std)
    
    ])
    test_image = transform(image).unsqueeze(0)  # إضافة batch dimension

    # inference
    with torch.no_grad():
        outputs = model(test_image)
        pred = nn.functional.softmax(outputs.data,dim=1)
        proba_pred = pred.max(dim=1)[0]
        pred_idx = pred.max(dim=1)[1]
        st.markdown(f"""
        <div style="
            background-color: #0B0D0F;  /* رمادي فاتح */
            border: 1px solid #ccc;      /* حدود بسيطة */
            border-radius: 12px;         /* زوايا مستديرة */
            padding: 20px;               /* مسافة داخل الكارد */
            text-align: center;          /* توسيط كل النصوص */
            max-width: 700px;            /* عرض أقصى */
            margin: auto;                /* توسيط الكارد داخل الصفحة */
        ">
            <h1>Result</h1>
            <h3>Status: {classes_map[pred_idx.item()]}</h3>
            <h4>Proba Status: {proba_pred.item():.2f}</h4>
        </div>
        """, unsafe_allow_html=True)
