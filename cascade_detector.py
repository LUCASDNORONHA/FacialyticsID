import cv2

class CascadeDetector:
    def __init__(self):
        """Carrega os classificadores Haarcascade."""
        self.frontalface_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")
        self.fullbody_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

    def detect_faces(self, gray):
        """Detecta rostos na imagem em escala de cinza."""
        return self.frontalface_cascade.detectMultiScale(gray, scaleFactor=1.02, minNeighbors=15, minSize=(130, 120), maxSize=(155, 155))

    def detect_smiles(self, roi_gray):
        """Detecta sorrisos na região do rosto."""
        return self.smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.02, minNeighbors=40, minSize=(75, 40), maxSize=(80, 45))
