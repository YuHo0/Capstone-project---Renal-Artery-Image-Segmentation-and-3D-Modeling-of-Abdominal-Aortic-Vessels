import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit

# 將包含 runUnet.py 的目錄新增到系統路徑
sys.path.append(r"D:\\GUI_code\\demo")

from runUnet import selectFiles, runUnet

class UNetWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Unet切割')
        layout = QVBoxLayout(self)

        self.selectButton = QPushButton('選擇檔案', self)
        self.selectButton.clicked.connect(self.updateFileName)  # 連接檔案選擇功能
        layout.addWidget(self.selectButton)

        # 文字框用於顯示檔案名稱
        self.fileNameTextBox = QLineEdit(self)
        layout.addWidget(self.fileNameTextBox)

        self.runButton = QPushButton('Unet', self)
        self.runButton.clicked.connect(self.runUnetWithSelectedPath)
        layout.addWidget(self.runButton)

    def updateFileName(self):
        # 選擇檔案並更新文字框中的檔案名稱
        selectedFileName = selectFiles()
        if selectedFileName:
            self.fileNameTextBox.setText(selectedFileName)

    def runUnetWithSelectedPath(self):
        selectedDirPath = self.fileNameTextBox.text()
        runUnet(selectedDirPath)

def main():
    app = QApplication(sys.argv)
    ex = UNetWindow()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
