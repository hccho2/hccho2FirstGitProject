## pytorch image load
```
import torchvision.models as models
import torchvision.transforms as transforms
import os
class MyVGG(nn.Module):
    def __init__(self,original_model):
        super(MyVGG, self).__init__()
        # vgg16에는 features, avgpool, classifier로 나누어져 있다.
        self.features = nn.Sequential(
            # features에서 마지막 3번재
            *list(original_model.features.children())[:-3]
        )
    def forward(self, x):
        x = self.features(x)
        return x 
os.environ['TORCH_HOME'] = './pretrained'
vgg16 = models.vgg16(pretrained=True, progress=True)  # 528M

transform=transforms.Compose([ transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
img = Image.open('dog.jpg')  # (576, 768, 3)
img = transform(img)  # torch.Size([3, 576, 768])

imgs = torch.unsqueeze(img,0)

feature1 = vgg16(imgs)  # ---> (N,1000)

extractor = MyVGG(vgg16)

featres2 = extractor(imgs)

```
