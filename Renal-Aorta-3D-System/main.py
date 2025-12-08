import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout

sys.path.append(r".\demo")

from function0_window import Function0Window
from function1_window import Function1Window
from function3_window import Function3Window
from unet_window import UNetWindow

class MenuWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('主選單')
        self.setGeometry(600, 600, 600, 400)

        centralWidget = QWidget(self)
        self.setCentralWidget(centralWidget)

        layout = QVBoxLayout(centralWidget)

        self.imageSegmentationButton = QPushButton('預處理', self)
        self.imageSegmentationButton.clicked.connect(self.openFunction0)
        layout.addWidget(self.imageSegmentationButton)

        self.spineRecognitionButton = QPushButton('脊椎辨識', self)
        self.spineRecognitionButton.clicked.connect(self.openFunction1)
        layout.addWidget(self.spineRecognitionButton)

        self.imageSegmentationButton = QPushButton('主動脈影像切割', self)
        self.imageSegmentationButton.clicked.connect(self.openUnet)
        layout.addWidget(self.imageSegmentationButton)

        self.imageSegmentationButton = QPushButton('3D建模', self)
        self.imageSegmentationButton.clicked.connect(self.openFunction3)
        layout.addWidget(self.imageSegmentationButton)

    def openFunction0(self):
        self.function2Window = Function0Window()
        self.function2Window.show()

    def openFunction1(self):
        self.function1Window = Function1Window()
        self.function1Window.show()

    def openUnet(self):
        self.function2Window = UNetWindow()
        self.function2Window.show()

    def openFunction3(self):
        self.function1Window = Function3Window()
        self.function1Window.show()

    

def main():
    app = QApplication(sys.argv)
    ex = MenuWindow()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()