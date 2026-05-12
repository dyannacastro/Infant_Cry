import os
import threading
import math
import numpy as np
import sounddevice as sd
import librosa
from joblib import load
from collections import Counter

from PyQt6.QtCore import Qt, QTimer, QRect, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap, QPainter, QColor, QFont
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QFrame, QMessageBox
)

# --- paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BG_IMAGE = os.path.join(SCRIPT_DIR, "imfant.jpg")
MODEL_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "infant_cry_svm_mfcc_pca_svm.joblib"),
    os.path.join(SCRIPT_DIR, "infant_cry_svm_mfcc.joblib"),
]

# ---------------- feature helpers (MODEL SIDE - UNTOUCHED) ----------------
def _stats_1d(v: np.ndarray) -> np.ndarray:
    p10 = np.percentile(v, 10); p90 = np.percentile(v, 90)
    return np.array([np.mean(v), np.std(v), np.median(v), p10, p90, p90 - p10], dtype=np.float32)

def _mfcc_block(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=cfg.get("n_mfcc", 40),
        n_mels=cfg.get("n_mels", 64),
        hop_length=cfg.get("hop_length", 512),
        n_fft=cfg.get("n_fft", 1024),
    )
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    def agg_rows(M):
        return np.concatenate([_stats_1d(M[i, :]) for i in range(M.shape[0])], axis=0)

    return np.concatenate([agg_rows(mfcc), agg_rows(d1), agg_rows(d2)], axis=0)

def _extras_block(y: np.ndarray, sr: int, cfg: dict) -> np.ndarray:
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr,
                                                  hop_length=cfg.get("hop_length", 512),
                                                  n_fft=cfg.get("n_fft", 1024))[0]
    spec_bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr,
                                                   hop_length=cfg.get("hop_length", 512),
                                                   n_fft=cfg.get("n_fft", 1024))[0]
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr,
                                                 hop_length=cfg.get("hop_length", 512),
                                                 n_fft=cfg.get("n_fft", 1024))[0]
    zcr       = librosa.feature.zero_crossing_rate(y,
                                                   frame_length=cfg.get("n_fft", 1024),
                                                   hop_length=cfg.get("hop_length", 512))[0]
    contrast  = librosa.feature.spectral_contrast(y=y, sr=sr,
                                                  hop_length=cfg.get("hop_length", 512),
                                                  n_fft=cfg.get("n_fft", 1024))
    parts = [
        _stats_1d(spec_cent),
        _stats_1d(spec_bw),
        _stats_1d(rolloff),
        _stats_1d(zcr),
        np.concatenate([_stats_1d(contrast[i, :]) for i in range(contrast.shape[0])], axis=0),
    ]
    return np.concatenate(parts, axis=0)

def _frame_audio(y: np.ndarray, sr: int, win_sec: float, hop_sec: float):
    win = int(round(win_sec * sr)); hop = int(round(hop_sec * sr))
    if len(y) < win:
        padded = np.zeros(win, dtype=y.dtype); padded[:len(y)] = y
        return [padded]
    return [y[i:i+win] for i in range(0, len(y) - win + 1, hop)]

def _expected_n_features_from_model(model) -> int | None:
    try:
        scaler = None
        if hasattr(model, "named_steps"):
            scaler = model.named_steps.get("scaler", None)
        if scaler is None and hasattr(model, "steps"):
            for name, step in model.steps:
                if name == "scaler":
                    scaler = step; break
        if scaler is None: return None
        if hasattr(scaler, "n_features_in_"): return int(scaler.n_features_in_)
        if hasattr(scaler, "mean_"): return int(len(scaler.mean_))
    except Exception:
        pass
    return None

# ---------------- PyQt helpers ----------------
class Bridge(QObject):
    status = pyqtSignal(str)
    file = pyqtSignal(str)
    prediction = pyqtSignal(str)
    set_mode = pyqtSignal(str)  # "ready" | "playing" | "predicting"
    error = pyqtSignal(str)

class AudioBars(QWidget):
    """Simple animated bars using QPainter (no external GIF)."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.setFixedSize(56, 28)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def start(self):
        if not self.timer.isActive():
            self.timer.start(60)

    def stop(self):
        self.timer.stop()
        self.update()

    def tick(self):
        self.phase += 0.35
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        bars = 5
        bar_w = 6
        gap = 6
        total = bars * bar_w + (bars - 1) * gap
        x0 = (w - total) // 2
        mid = h // 2

        color = QColor(20, 20, 20, 220)
        p.setBrush(color)
        p.setPen(Qt.PenStyle.NoPen)

        for i in range(bars):
            amp = (math.sin(self.phase + i * 0.9) * 0.5 + 0.5)  # 0..1
            bh = int(6 + amp * (h - 8))
            x = x0 + i * (bar_w + gap)
            rect = QRect(x, mid - bh // 2, bar_w, bh)
            p.drawRoundedRect(rect, 3, 3)

class GlassCard(QFrame):
    """Real translucent card."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # shadow
        shadow = QColor(0, 0, 0, 60)
        p.setBrush(shadow)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(self.rect().adjusted(6, 6, -2, -2), 18, 18)

        # glass
        glass = QColor(255, 255, 255, 140)  # <-- real alpha
        border = QColor(255, 255, 255, 160)
        p.setBrush(glass)
        p.setPen(border)
        p.drawRoundedRect(self.rect().adjusted(0, 0, -8, -8), 18, 18)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Infant Cry Classifier")
        self.resize(960, 540)

        # for real transparency on child widgets
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        self.bg = QPixmap(BG_IMAGE) if os.path.isfile(BG_IMAGE) else QPixmap()
        if self.bg.isNull():
            self.setStyleSheet("background: #F2F4F7;")
        else:
            self.setStyleSheet("background: black;")  # ignored by paintEvent

        # model state
        self.model = None
        self.class_names = None
        self.cfg = None
        self.uses_windows = False
        self.include_extras = False

        self.bridge = Bridge()
        self.bridge.status.connect(self.on_status)
        self.bridge.file.connect(self.on_file)
        self.bridge.prediction.connect(self.on_prediction)
        self.bridge.set_mode.connect(self.on_mode)
        self.bridge.error.connect(self.on_error)

        # UI layout
        self.card = GlassCard(self)
        self.card.setFixedHeight(210)
        self.card.setFixedWidth(520)


        self.title = QLabel("Infant Cry Classifier")
        self.title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))

        self.status = QLabel("Ready")
        self.status.setFont(QFont("Segoe UI", 10))
        self.status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        header = QHBoxLayout()
        header.addWidget(self.title)
        header.addStretch(1)
        header.addWidget(self.status)

        self.pred_label = QLabel("Prediction")
        self.pred_label.setFont(QFont("Segoe UI", 10))

        self.pred_value = QLabel("—")
        self.pred_value.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))

        self.file_label = QLabel("File")
        self.file_label.setFont(QFont("Segoe UI", 10))

        self.file_value = QLabel("No file selected")
        self.file_value.setFont(QFont("Segoe UI", 10))
        self.file_value.setWordWrap(True)

        row = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(self.pred_label)
        left.addWidget(self.pred_value)
        row.addLayout(left, 1)

        right = QVBoxLayout()
        right.addWidget(self.file_label)
        right.addWidget(self.file_value)
        row.addLayout(right, 1)

        self.btn = QPushButton("Upload Audio")
        self.btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn.setMinimumHeight(44)

        self.btn.setStyleSheet("""
        QPushButton {
            background-color: #2563EB;      /* primary blue */
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: 12px;
            font-size: 14px;
            font-weight: 600;
        }
        QPushButton:hover {
            background-color: #1D4ED8;      /* darker on hover */
        }
        QPushButton:pressed {
            background-color: #1E40AF;      /* darkest on press */
        }
        QPushButton:disabled {
            background-color: rgba(37, 99, 235, 90);
            color: rgba(255, 255, 255, 190);
        }   
    """)

        self.btn.clicked.connect(self.choose_audio)

        self.bars = AudioBars()
        self.bars.hide()
        self.anim_text = QLabel("")
        self.anim_text.setFont(QFont("Segoe UI", 10))

        bottom = QHBoxLayout()
        bottom.addWidget(self.btn)
        bottom.addStretch(1)
        bottom.addWidget(self.bars)
        bottom.addWidget(self.anim_text)

        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(22, 18, 22, 18)
        card_layout.addLayout(header)
        card_layout.addSpacing(8)
        card_layout.addLayout(row)
        card_layout.addStretch(1)
        card_layout.addLayout(bottom)

        # root layout (position card near bottom)
        root = QVBoxLayout(self)
        root.addStretch(1)
        root.addWidget(self.card, alignment=Qt.AlignmentFlag.AlignHCenter)
        root.setContentsMargins(24, 24, 24, 24)

        # autoload model
        self.btn.setEnabled(False)
        self._autoload_model_or_prompt()
        self.btn.setEnabled(True)

    def paintEvent(self, e):
        if self.bg.isNull():
            return super().paintEvent(e)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawPixmap(self.rect(), self.bg)

    # ---- UI slots ----
    def on_status(self, s): self.status.setText(s)
    def on_file(self, s): self.file_value.setText(s)
    def on_prediction(self, s): self.pred_value.setText(s)

    def on_mode(self, mode):
        if mode == "ready":
            self.status.setText("Ready")
            self.anim_text.setText("")
            self.bars.stop()
            self.bars.hide()
            self.btn.setEnabled(True)
        elif mode == "playing":
            self.status.setText("Playing audio…")
            self.anim_text.setText("Playing audio…")
            self.bars.show()
            self.bars.start()
            self.btn.setEnabled(False)
        elif mode == "predicting":
            self.status.setText("Predicting…")
            self.anim_text.setText("Analyzing…")
            self.bars.stop()
            self.bars.show()
            self.btn.setEnabled(False)

    def on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)

    # ---- model loading (MODEL SIDE - UNTOUCHED) ----
    def _autoload_model_or_prompt(self):
        for cand in MODEL_CANDIDATES:
            if os.path.isfile(cand) and self._load_model_from_path(cand):
                return
        path, _ = QFileDialog.getOpenFileName(self, "Select trained model (.joblib)", SCRIPT_DIR, "Joblib (*.joblib);;All Files (*)")
        if path and self._load_model_from_path(path):
            return
        QMessageBox.critical(self, "Model not found", "Place your trained .joblib next to this app and restart.")
        raise SystemExit(1)

    def _load_model_from_path(self, path: str) -> bool:
        try:
            payload = load(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")
            return False
        if not all(k in payload for k in ["model", "class_names", "feature_config"]):
            QMessageBox.critical(self, "Error", "Selected file is not a compatible model payload.")
            return False

        self.model = payload["model"]
        self.class_names = payload["class_names"]
        self.cfg = payload["feature_config"]
        self.uses_windows = ("win_sec" in self.cfg and "hop_sec" in self.cfg)

        base_dim = int(self.cfg.get("n_mfcc", 20)) * 3 * 6
        expected = _expected_n_features_from_model(self.model)
        self.include_extras = (expected is not None and expected > base_dim)
        return True

    # ---- play first then predict ----
    def choose_audio(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select audio file", SCRIPT_DIR, "Audio (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*)")
        if not path:
            return
        self.bridge.file.emit(os.path.basename(path))
        self.bridge.prediction.emit("—")
        threading.Thread(target=self._play_then_predict_worker, args=(path,), daemon=True).start()

    def _play_then_predict_worker(self, path: str):
        try:
            self.bridge.set_mode.emit("playing")
            y_play, sr_play = librosa.load(path, sr=None, mono=True)
            try: sd.stop()
            except Exception: pass
            sd.play(y_play, sr_play)
            sd.wait()

            self.bridge.set_mode.emit("predicting")
            result = self._predict_path(path)
            self.bridge.prediction.emit(result)

        except Exception as e:
            self.bridge.error.emit(f"Failed to process audio:\n{e}")
        finally:
            self.bridge.set_mode.emit("ready")

    # ---------------- prediction logic (MODEL SIDE - UNTOUCHED) ----------------
    def _features_for_window(self, y: np.ndarray, sr: int) -> np.ndarray:
        feat = _mfcc_block(y, sr, self.cfg)
        if self.include_extras:
            feat = np.concatenate([feat, _extras_block(y, sr, self.cfg)], axis=0)
        return feat.astype(np.float32)

    def _predict_path(self, path: str) -> str:
        sr = int(self.cfg.get("sr", 16000))
        y, sr = librosa.load(path, sr=sr, mono=True)

        if self.uses_windows:
            if "trim_top_db" in self.cfg and self.cfg["trim_top_db"] is not None:
                y, _ = librosa.effects.trim(y, top_db=self.cfg["trim_top_db"])
                if y.size == 0:
                    y = np.zeros(int(round(self.cfg.get("win_sec", 2.0) * sr)), dtype=np.float32)

            frames = _frame_audio(y, sr, self.cfg.get("win_sec", 2.0), self.cfg.get("hop_sec", 1.0))
            feats = [self._features_for_window(f, sr) for f in frames]
            X = np.stack(feats)
            preds = self.model.predict(X)
            final_idx = Counter(preds.tolist()).most_common(1)[0][0]
            return self.class_names[int(final_idx)]
        else:
            feat = self._features_for_window(y, sr).reshape(1, -1)
            pred = int(self.model.predict(feat)[0])
            return self.class_names[pred]


def main():
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()

if __name__ == "__main__":
    main()
