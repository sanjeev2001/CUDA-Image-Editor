import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QFileDialog, QVBoxLayout, QPushButton, QWidget
from PyQt6.QtGui import QPixmap, QImage, QColor
from PyQt6.QtCore import Qt
import numpy as np
import cupy as cp
import time

class ImageDisplayApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        # Create central widget and set layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create QLabel to display the image
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        # Create 'Open Image' button
        open_button = QPushButton('Open Image', self)
        open_button.clicked.connect(self.openImage)
        layout.addWidget(open_button)

        # Create 'Grayscale' button
        grayscale_button = QPushButton('Grayscale', self)
        grayscale_button.clicked.connect(self.applyGrayscale)
        layout.addWidget(grayscale_button)

        # Create 'Sepia' button
        sepia_button = QPushButton('Sepia', self)
        sepia_button.clicked.connect(self.applySepia)
        layout.addWidget(sepia_button)

        # Set up the main window
        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Image Display App')

        self.pixmap = None

    def openImage(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image Files (*.png *.jpg *.bmp *.gif *.jpeg)")
        
        if fileName:
            self.pixmap = QPixmap(fileName)
            self.image_label.setPixmap(self.pixmap.scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.image_label.adjustSize()

    def rgbToGray(self, image):
        gray_gpu = cp.dot(image[..., :3], cp.asarray([0.299, 0.587, 0.114]))
        # Normalize and cast to uint8
        gray_gpu = cp.clip(gray_gpu, 0, 255).astype(cp.uint8)
        gray_cpu = cp.asnumpy(gray_gpu)
        
        return gray_cpu

    def applyGrayscale(self):
        if self.pixmap:
            # CUDA implementation ==> 1.9970204830169678 seconds for 8k image
            image = self.pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
            width = image.width()
            height = image.height()
            ptr = image.constBits()
            ptr.setsize(height * width * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
            img_gpu = cp.array(arr, dtype=cp.float32)
            gray_array = self.rgbToGray(img_gpu)
            gray_image = QImage(gray_array.data, width, height, gray_array.strides[0], QImage.Format.Format_Grayscale8)
            
            # cpu implementation ==> 88.46851062774658 seconds for 8k image
            # for x in range(width):
            #     for y in range(height):
            #         color = QColor(image.pixel(x, y))
            #         gray_value = int(color.red() * 0.299) + int(color.green() * 0.587) + int(color.blue() * 0.114)
            #         gray_color = QColor(gray_value, gray_value, gray_value)
            #         image.setPixelColor(x, y, gray_color)

            self.image_label.setPixmap(QPixmap.fromImage(gray_image).scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.image_label.adjustSize()
            
            # 44.3x increase in speed using CUDA

    def rgbToSep(self, image):
        sepia_filter = cp.array([[0.393, 0.769, 0.189],
                             [0.349, 0.686, 0.168],
                             [0.272, 0.534, 0.131]])
        image_array_gpu = cp.asarray(image)
        sepia_result_gpu = cp.dot(image_array_gpu.reshape(-1, 3), sepia_filter.T).reshape(image.shape)

        # Clip the values to be in the valid range [0, 255]
        sepia_result_gpu = cp.clip(sepia_result_gpu, 0, 255).astype(cp.uint8)
        
        # Step 3: Convert NumPy array back to QImage
        sepia_result_cpu = cp.asnumpy(sepia_result_gpu)

        return sepia_result_cpu

    def applySepia(self):
        if self.pixmap:
            image = self.pixmap.toImage().convertToFormat(QImage.Format.Format_RGB32)
            width = image.width()
            height = image.height()
            ptr = image.constBits()
            ptr.setsize(height * width * 4)
            arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
            img_gpu = cp.array(arr, dtype=cp.float32)
            sepia_array = self.rgbToSep(img_gpu)
            sepia_image = QImage(sepia_array.data, width, height, sepia_array.strides[0], QImage.Format.Format_RGB32)
            # for x in range(width):
            #     for y in range(height):
            #         color = QColor(image.pixel(x, y))
            #         sr = min(int(0.393 * color.red() + 0.769 * color.green() + 0.189 * color.blue()), 255)
            #         sg = min(int(0.349 * color.red() + 0.686 * color.green() + 0.168 * color.blue()), 255)
            #         sb = min(int(0.272 * color.red() + 0.534 * color.green() + 0.131 * color.blue()), 255)
            #         sepia_color = QColor(sr, sg, sb, color.alpha())
            #         image.setPixelColor(x, y, sepia_color)
            
            self.image_label.setPixmap(QPixmap.fromImage(sepia_image).scaled(self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            self.image_label.adjustSize()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = ImageDisplayApp()
    mainWindow.show()
    sys.exit(app.exec())