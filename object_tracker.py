import cv2

class ObjectTracker:
    def __init__(self):
        self.padding = 20  # Aumenta o tamanho do quadrado ao redor do rosto.

    def draw_faces(self, frame, faces, smile_cascade):
        """Desenha retângulos ao redor das faces e detecta sorrisos."""
        for (x, y, w, h) in faces:
            x1 = max(0, x - self.padding)
            y1 = max(0, y - self.padding)
            x2 = min(frame.shape[1], x + w + self.padding)
            y2 = min(frame.shape[0], y + h + self.padding)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
            cv2.putText(frame, 'Face', (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 0.5, (255, 255, 0), 1)

            roi_gray = frame[y: y + h, x: x + w]
            roi_color = frame[y: y + h, x: x + w]

            smiles = smile_cascade.detect_smiles(roi_gray)

            for (sx, sy, sw, sh) in smiles:
                cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
                cv2.putText(roi_color, 'Smile', (sx, sy - 10), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
