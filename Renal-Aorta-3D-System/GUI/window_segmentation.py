import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout
from unet_window import UNetWindow  # 假設你已經有了 unet_window.py 的定義
from unet_renal_window import UNetrenalWindow  # 假設你已經有了 unet_renal_window.py 的定義

class Function2Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(600, 600, 600, 400)
        self.setWindowTitle('Unet入口')
        self.setGeometry(100, 100, 300, 200)

        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)

        layout = QVBoxLayout(centralWidget)

        self.unetButton = QPushButton('UNet', self)
        self.unetButton.clicked.connect(self.openUNetWindow)
        layout.addWidget(self.unetButton)

        self.unetRenalButton = QPushButton('UNet_Renal', self)
        self.unetRenalButton.clicked.connect(self.openUNetRenalWindow)
        layout.addWidget(self.unetRenalButton)

    def openUNetWindow(self):
        self.unetWindow = UNetWindow()
        self.unetWindow.show()

    def openUNetRenalWindow(self):
        self.unetRenalWindow = UNetrenalWindow()
        self.unetRenalWindow.show()

def main():
    app = QApplication(sys.argv)
    ex = Function2Window()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
