import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QFileDialog, QLabel, QMessageBox
import os

# 添加包含 function3_process 的目錄到系統路徑
sys.path.append("D:\\GUI_code\\demo\\3D")
from function3_process import processImages  # 正確導入

class Function3Window(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(600, 600, 600, 400)
        self.setWindowTitle('3D建模')
        layout = QVBoxLayout(self)

        # 初始化 input_folder_path
        self.input_folder_path = ""

        # 按鈕用於選擇輸入文件夾
        self.inputFolderButton = QPushButton('選擇輸入文件夾', self)
        self.inputFolderButton.clicked.connect(self.updateInputFolder)
        layout.addWidget(self.inputFolderButton)

        # 文字框用於顯示檔案名稱
        self.fileNameTextBox = QLineEdit(self)
        layout.addWidget(self.fileNameTextBox)

        # 按鈕用於開始處理
        self.processButton = QPushButton('開始處理', self)
        self.processButton.clicked.connect(self.processImages)
        layout.addWidget(self.processButton)

    def updateInputFolder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "選擇輸入文件夾")
        if folder_path:
            self.fileNameTextBox.setText(folder_path)
            self.input_folder_path = folder_path

    def processImages(self):
        if not self.input_folder_path:
            QMessageBox.warning(self, "警告", "請選擇輸入文件夾")
            return
        dirPath = self.input_folder_path + "\\*.png"
        output_dir = os.path.join(self.input_folder_path, '3d_output')
        processImages(dirPath, output_dir)  # 傳遞輸出目錄參數
        QMessageBox.information(self, "完成", "3D建模完成，文件已保存到: " + output_dir)

def main():
    app = QApplication(sys.argv)
    ex = Function3Window()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
