from camera import Camera
from cascade_detector import CascadeDetector
from object_tracker import ObjectTracker
import cv2

class App:
    def __init__(self):
        self.camera = Camera()
        self.detector = CascadeDetector()
        self.tracker = ObjectTracker()

    def run(self):
        """Executa o loop principal do aplicativo."""
        while self.camera.cap.isOpened():
            ret, frame = self.camera.read_frame()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = self.detector.detect_faces(gray)

            self.tracker.draw_faces(frame, faces, self.detector)

            cv2.imshow('Rastreamento de Pessoas', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = App()
    app.run()