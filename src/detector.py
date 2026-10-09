"""Single-mouse detector: fine-tuned YOLO11n run on the arena crop."""
from ultralytics import YOLO


class MouseDetector:
    def __init__(self, weights, calibration, imgsz=352, conf=0.25):
        self.model = YOLO(weights)
        self.calib = calibration
        self.imgsz = imgsz
        self.conf = conf
        self._crop = None

    def detect_batch(self, frames):
        """List of BGR frames -> list of (cx, cy, conf) in frame pixels, or None if no mouse found."""
        if self._crop is None:
            self._crop = self.calib.crop_box(frames[0].shape)
        x1, y1, x2, y2 = self._crop
        results = self.model.predict([f[y1:y2, x1:x2] for f in frames], imgsz=self.imgsz, conf=self.conf,
                                     verbose=False)
        out = []
        for r in results:
            if not len(r.boxes):
                out.append(None)
                continue
            j = int(r.boxes.conf.argmax())  # one mouse: keep the most confident box
            bx1, by1, bx2, by2 = r.boxes.xyxy[j].tolist()
            out.append(((bx1 + bx2) / 2 + x1, (by1 + by2) / 2 + y1, float(r.boxes.conf[j])))
        return out
