from torch import nn
import  torchvision
from torchvision.models import mobilenet_v2

model_mobilenetv2 = mobilenet_v2(weights="DEFAULT")

model_mobilenetv2.classifier = nn.Identity()

class MobileNetV2_Alzheimer_Classfier(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self._model = nn.Sequential(
            model_mobilenetv2,
            nn.Linear(in_features=1280,out_features=1000,bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(in_features =1000,out_features=num_classes,bias=True)
       
        )

    def forward(self,x):
        return self._model(x)


        