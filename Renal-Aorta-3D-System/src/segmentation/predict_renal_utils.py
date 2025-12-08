import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from unet_model import UNet_ResNet

# 定義常數
DATA_PATH = 'D:\\GUI_code\\demo\\unet_renal\\input\\'
TEST_IMAGE_PATH = r'test/images/'
TEST_MASK_PATH = r'test/masks/'
IMAGE_TYPE = '.bmp'
MASK_TYPE = '.png'
IMG_HEIGHT = 256
IMG_WIDTH = 256
REAL_HEIGHT = 512
REAL_WIDTH = 512
BATCH_SIZE = 8
MODEL_PATH = r'D:\\GUI_code\\demo\\unet_renal\\unetv1_renal.pth'
N_CLASSES = 1
SAVE_RESULTS_PATH = r'D:\\GUI_code\\demo\\unet_renal\\preds\\'

# GPU 可用的話使用 GPU，否則使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定義測試資料集類別
class Test_PotholeDataset(Dataset):
    def __init__(self, root_dir=DATA_PATH, transform=None):
        self.root_dir = root_dir
        listname = []
        for imgfile in os.listdir(DATA_PATH + TEST_MASK_PATH):
            if os.path.splitext(imgfile)[1] == MASK_TYPE:
                filename = os.path.splitext(imgfile)[0]
                listname.append(filename)
        self.ids = listname
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        id = self.ids[index]
        image = Image.open(self.root_dir + TEST_IMAGE_PATH + id + IMAGE_TYPE)
        image = self.transform(image)
        return image

# 載入預訓練模型
model = UNet_ResNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# 載入測試資料集
test_dataset = Test_PotholeDataset(DATA_PATH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 預測並顯示結果
def predict(model, test_loader, device):
    model.eval()
    predicted_masks = []
    back_transform = transforms.Compose([transforms.Resize((REAL_HEIGHT, REAL_WIDTH))])
    with torch.no_grad():
        for input in test_loader:
            input = input.to(device)
            predict = model(input)
            predict = back_transform(predict)
            predict = (predict > 0).type(torch.float)
            predicted_masks.append(predict)
    predicted_masks = torch.cat(predicted_masks)
    return predicted_masks

predicted_mask = predict(model, test_loader, device=device)

import cv2
print (cv2.__version__)

for i in range(210):
    n_samples=i
    
    sample = predicted_mask[i]  
    sample = torch.squeeze(sample, dim=0)
    sample = transforms.ToPILImage()(sample)

    #plt.imshow(sample)
    #plt.savefig('./preds/predict_'+str(n_samples)+'.png')
    plt.imsave('D:\\GUI_code\\demo\\unet_renal\\preds\\predict_'+str(n_samples)+'.png',sample, cmap='gray')

# 顯示結果
def show_sample_test_result(test_dataset, predicted_mask, n_samples=10):
    """Visualize test sample and corresponding result."""
    plt.rcParams["figure.figsize"] = (30, 15)
    back_transform = transforms.Compose([transforms.Resize((REAL_HEIGHT, REAL_WIDTH))])
    for i in range(n_samples):
        sample = predicted_mask[i]
        sample = torch.squeeze(sample, dim=0)
        sample = transforms.ToPILImage()(sample)
        X = test_dataset[i]
        X = back_transform(X)
        X = transforms.ToPILImage()(X)

        ax = plt.subplot(2, int(n_samples / 2), i + 1)
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        plt.imshow(X, cmap="Greys")
        plt.imshow(sample, alpha=0.3, cmap="OrRd")
        if i == n_samples - 1:
            plt.show()
            break

show_sample_test_result(test_dataset, predicted_mask)
