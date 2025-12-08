import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from unet_model import UNet_ResNet

# 定義常數
MODEL_PATH = 'D:\\GUI_code\\demo\\unet\\unetv1_renal.pth'
IMG_HEIGHT = 256
IMG_WIDTH = 256
REAL_HEIGHT = 512
REAL_WIDTH = 512
BATCH_SIZE = 8  # 批次大小
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入預訓練模型
model = UNet_ResNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# 定義預處理和後處理變換
preprocess = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Grayscale(num_output_channels=3),  # 將灰度圖像轉換為3通道圖像
    transforms.ToTensor(),
])

back_transform = transforms.Compose([
    transforms.Resize((REAL_HEIGHT, REAL_WIDTH))
])

def predict_batch(images, model, device):
    # 將圖像批次移動到GPU/CPU
    images = torch.stack(images).to(device)

    # 進行預測
    with torch.no_grad():
        predictions = model(images)
        predictions = torch.stack([back_transform(pred) for pred in predictions])
        predictions = (predictions > 0).type(torch.float)
    
    # 去除批次維度並轉換為PIL圖像
    predictions = [transforms.ToPILImage()(torch.squeeze(pred, dim=0)) for pred in predictions]
    
    return predictions

def predict_folder(image_folder_path, output_folder_path, model, device, batch_size=BATCH_SIZE):
    # 確保輸出資料夾存在
    os.makedirs(output_folder_path, exist_ok=True)
    
    # 獲取資料夾中的所有圖像文件
    image_files = [f for f in os.listdir(image_folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    images = []
    file_paths = []

    for image_file in image_files:
        image_path = os.path.join(image_folder_path, image_file)
        image = Image.open(image_path)
        image = preprocess(image)
        images.append(image)
        file_paths.append(image_file)
        
        # 當累積的圖像數量達到批次大小時進行預測
        if len(images) == batch_size:
            predictions = predict_batch(images, model, device)
            for pred, file_path in zip(predictions, file_paths):
                output_path = os.path.join(output_folder_path, f"predicted_{file_path}")
                pred.save(output_path)
                plt.imshow(pred, cmap='gray')
            images = []
            file_paths = []
    
    # 處理剩餘的圖像
    if images:
        predictions = predict_batch(images, model, device)
        for pred, file_path in zip(predictions, file_paths):
            output_path = os.path.join(output_folder_path, f"predicted_{file_path}")
            pred.save(output_path)
            plt.imshow(pred, cmap='gray')
            

# 設定資料夾路徑
image_folder_path = 'D:\\GUI_code\\data\\unet_original\\patient_01'
output_folder_path = 'D:\\GUI_code\\data\\unet\\patient_01_prediction'

# 預測整個資料夾
predict_folder(image_folder_path, output_folder_path, model, device)

