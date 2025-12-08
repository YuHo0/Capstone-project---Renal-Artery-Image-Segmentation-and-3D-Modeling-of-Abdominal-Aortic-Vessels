import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QMessageBox

# 將包含 runYOLO.py 的目錄新增到系統路徑
sys.path.append("D:\\GUI_code\\demo")

from runYOLO import selectFiles, runYolo

class Function1Window(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.selectedFileName = ""  # 新增變數來保存選擇的檔案名稱

    def initUI(self):
        self.setGeometry(600, 600, 600, 400)
        self.setWindowTitle('YOLO')
        layout = QVBoxLayout(self)

        self.selectButton = QPushButton('選擇檔案', self)
        self.selectButton.clicked.connect(self.updateFileName)  # 連接檔案選擇功能
        layout.addWidget(self.selectButton)

        # 文字框用於顯示檔案名稱
        self.fileNameTextBox = QLineEdit(self)
        layout.addWidget(self.fileNameTextBox)

        self.runButton = QPushButton('YOLO v4 Tiny', self)
        self.runButton.clicked.connect(self.runYoloWithSelectedFile)
        layout.addWidget(self.runButton)

    def updateFileName(self):
        # 選擇檔案並更新文字框中的檔案名稱
        selectedFileName = selectFiles()
        if selectedFileName:
            self.selectedFileName = selectedFileName
            self.fileNameTextBox.setText(self.selectedFileName)
    
    def runYoloWithSelectedFile(self):
        if self.selectedFileName:
            runYolo(self.selectedFileName)
        else:
            QMessageBox.critical(self, "錯誤", "請先選擇檔案。")

def main():
    app = QApplication(sys.argv)
    ex = Function1Window()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
