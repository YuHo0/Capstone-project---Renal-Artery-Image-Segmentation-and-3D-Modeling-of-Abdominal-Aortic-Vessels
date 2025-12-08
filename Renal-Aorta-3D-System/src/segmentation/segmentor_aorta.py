import os
import subprocess
from PyQt5.QtWidgets import QFileDialog, QMessageBox

# 全域變數，用於存儲所選的檔案名稱
selectedTxtFileName = ""

def selectFiles():
    # 設置檔案選擇對話框的選項
    options = QFileDialog.Options()
    options |= QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks

    # 開啟資料夾選擇對話框，允許用戶選擇一個資料夾
    global selectedDirPath
    selectedDirPath = QFileDialog.getExistingDirectory(None, "選擇資料夾", "", options=options)
    
    return selectedDirPath

# 明确设置工作目录
os.chdir(r'D:\\GUI_code\\demo')

def runUnet(): 
    # 執行 UNet 相關的功能
    unet_script_path = r".\\unet\\unet_predict.py"
    subprocess.Popen(["python", unet_script_path, selectedDirPath])

if __name__ == '__main__':
    runUnet()
