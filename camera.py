import cv2

class Camera:
    def __init__(self, index=0):
        """Inicializa a captura da webcam."""
        self.cap = cv2.VideoCapture(index)

    def read_frame(self):
        """Lê um frame da câmera."""
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """Libera a câmera."""
        self.cap.release()
