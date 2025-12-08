import os
import subprocess
from PyQt5.QtWidgets import QFileDialog, QMessageBox

# 全域變數，用於存儲所選的檔案名稱
selectedTxtFileName = ""

def selectFiles():
    # 設置檔案選擇對話框的選項
    options = QFileDialog.Options()
    
    # 開啟檔案選擇對話框，允許用戶選擇多個檔案
    global selectedTxtFileName
    files, _ = QFileDialog.getOpenFileNames(None, "選擇檔案", "", "Files (*.txt)", options=options)
    
    if files:
        # 這裡我們假設只選擇一個檔案，因此取第一個檔案的檔案名稱
        selectedTxtFileName = os.path.basename(files[0])
        return selectedTxtFileName

def runYolo():
    global selectedTxtFileName
    if not selectedTxtFileName:
        QMessageBox.critical(None, "錯誤", "請先選擇檔案。")
        return

    yolo_dir = r".\v4tiny"

    if not os.path.exists(yolo_dir):
        QMessageBox.critical(None, "錯誤", f"目錄 {yolo_dir} 不存在。")
        return

    selectedTxtFile = os.path.join(selectedTxtFileName)

    os.chdir(yolo_dir)

    command = (
        f"darknet detector test data/obj.data yolov4-tiny-custom.cfg "
        f"backup/yolov4-tiny-custom_final.weights -dont_show -ext_output "
        f"< patients/{selectedTxtFile} > patients_result/result.txt"
    )
    try:
        subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        QMessageBox.information(None, "YOLO 執行", "YOLO 已完成處理。請查看 patients_result/result.txt 獲取輸出。")
    except subprocess.CalledProcessError as e:
        QMessageBox.critical(None, "錯誤", f"執行 YOLO 時發生錯誤：\n{e.stderr}")
    except Exception as e:
        QMessageBox.critical(None, "錯誤", f"發生意外錯誤：{e}")
