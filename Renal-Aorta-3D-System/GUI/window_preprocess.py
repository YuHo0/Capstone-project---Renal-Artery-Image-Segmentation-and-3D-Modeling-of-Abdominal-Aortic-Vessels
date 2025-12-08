import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QFileDialog, QMessageBox, QLabel

# 添加包含 combine.py 的目錄到系統路徑
sys.path.append("D:/GUI_code/demo/preprocessing")
from combine import process_folder

class Function0Window(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(600, 600, 600, 400)
        self.setWindowTitle('預處理影像')
        layout = QVBoxLayout(self)

        # 按鈕用於選擇輸入資料夾
        self.inputFolderButton = QPushButton('選擇輸入資料夾', self)
        self.inputFolderButton.clicked.connect(self.updateInputFolderName)
        layout.addWidget(self.inputFolderButton)

        # 創建一個水平佈局來包含標籤和文字框
        inputFolderLayout = QHBoxLayout()
        
        # 添加標籤 "輸入資料夾:"
        self.inputFolderLabel = QLabel('輸入資料夾:', self)
        inputFolderLayout.addWidget(self.inputFolderLabel)

        # 文字框用於顯示選擇的輸入資料夾路徑
        self.inputFolderTextBox = QLineEdit(self)
        inputFolderLayout.addWidget(self.inputFolderTextBox)

        # 將水平佈局添加到主佈局中
        layout.addLayout(inputFolderLayout)

        # 創建一個水平佈局來包含標籤和文字框
        outputFolderLayout = QHBoxLayout()

        # 添加標籤 "輸出資料夾:"
        self.outputFolderLabel = QLabel('輸出資料夾:', self)
        outputFolderLayout.addWidget(self.outputFolderLabel)

        # 文字框用於顯示選擇的輸出資料夾路徑（自動設置）
        self.outputFolderTextBox = QLineEdit(self)
        self.outputFolderTextBox.setReadOnly(True)
        outputFolderLayout.addWidget(self.outputFolderTextBox)

        # 將水平佈局添加到主佈局中
        layout.addLayout(outputFolderLayout)

        # 按鈕用於開始處理影像
        self.processButton = QPushButton('處理影像', self)
        self.processButton.clicked.connect(self.processImages)
        layout.addWidget(self.processButton)

    def updateInputFolderName(self):
        folder_path = QFileDialog.getExistingDirectory(self, "選擇輸入資料夾")
        if folder_path:
            self.inputFolderTextBox.setText(folder_path)
            # 自動設置輸出資料夾路徑
            output_folder = os.path.join("D:/GUI_code/data/output", os.path.basename(folder_path))
            self.outputFolderTextBox.setText(output_folder)

    def processImages(self):
        input_folder = self.inputFolderTextBox.text()
        output_folder = self.outputFolderTextBox.text()

        if not input_folder or not os.path.isdir(input_folder):
            QMessageBox.critical(self, "錯誤", "請選擇有效的輸入資料夾")
            return

        try:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            process_folder(input_folder, output_folder)
            save_bmp_file_paths(output_folder, os.path.join(output_folder, os.path.basename(output_folder) + '.txt'))
            QMessageBox.information(self, "成功", "影像處理完成，並保存檔案路徑")
        except Exception as e:
            QMessageBox.critical(self, "錯誤", f"處理影像時發生錯誤: {str(e)}")

def save_bmp_file_paths(directory, output_file):
    # 打開輸出的文件，以寫入模式
    with open(output_file, 'w', encoding='utf-8') as file:
        # 遍歷指定資料夾中的所有文件和子資料夾
        for root, dirs, files in os.walk(directory):
            for name in files:
                # 只處理 .bmp 文件
                if name.endswith('.bmp'):
                    # 獲取文件的完整路徑
                    file_path = os.path.join(root, name)
                    # 將文件路徑寫入txt檔中
                    file.write(file_path + '\n')

def main():
    app = QApplication(sys.argv)
    ex = Function0Window()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
