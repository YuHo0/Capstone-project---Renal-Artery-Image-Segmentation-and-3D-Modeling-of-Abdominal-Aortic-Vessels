import os
import sys
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from unet_model import UNet_ResNet
import tkinter as tk
from tkinter import messagebox

# 確保從命令列獲取資料夾路徑
if len(sys.argv) < 2:
    print("使用方法: python script.py <圖片資料夾路徑>")
    sys.exit(1)

# 取得資料夾路徑參數
DATA_PATH = sys.argv[1]

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
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 定義預處理和後處理變換
preprocess = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.Grayscale(num_output_channels=3),  # 將灰度圖像轉換為3通道圖像
    transforms.ToTensor(),
])

back_transform = transforms.Compose([
    transforms.Resize((REAL_HEIGHT, REAL_WIDTH)),
    transforms.ToPILImage()
])

# 創建輸出資料夾
output_folder_name = os.path.basename(DATA_PATH)
output_folder_path = os.path.join('D:\\GUI_code\\data\\3d', output_folder_name)
os.makedirs(output_folder_path, exist_ok=True)

def predict_batch(images, model, device):
    # 將圖像批次移動到GPU/CPU
    images = torch.stack(images).to(device)

    # 進行預測
    with torch.no_grad():
        predictions = model(images)
        predictions = (predictions > 0).type(torch.float)

    # 去除批次維度並轉換為PIL圖像
    predictions = [transforms.ToPILImage()(torch.squeeze(pred.cpu(), dim=0)) for pred in predictions]
    
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
            images = []
            file_paths = []
    
    # 處理剩餘的圖像
    if images:
        predictions = predict_batch(images, model, device)
        for pred, file_path in zip(predictions, file_paths):
            output_path = os.path.join(output_folder_path, f"predicted_{file_path}")
            pred.save(output_path)

if __name__ == "__main__":
    predict_folder(DATA_PATH, output_folder_path, model, device)

# 創建一個提示窗口
root = tk.Tk()
root.withdraw()  # 隱藏主窗口

# 顯示消息框
messagebox.showinfo("提示", "程式已執行完畢！")

# 主窗口事件循環
root.mainloop()