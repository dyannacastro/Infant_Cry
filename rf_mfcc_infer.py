import sys
import numpy as np
import librosa
from joblib import load
from collections import Counter

MODEL_PATH = "infant_cry_rf_mfcc_pca.joblib"  # example only

def frame_audio(y, sr, win_sec, hop_sec):
    win = int(round(win_sec * sr))
    hop = int(round(hop_sec * sr))
    if len(y) < win:
        out = np.zeros(win, dtype=y.dtype)
        out[:len(y)] = y
        return [out]
    return [y[i:i+win] for i in range(0, len(y) - win + 1, hop)]

def stats_1d(v):
    p10 = np.percentile(v, 10); p90 = np.percentile(v, 90)
    return np.array([np.mean(v), np.std(v), np.median(v), p10, p90, p90 - p10], dtype=np.float32)

def extract_features_window_mfcc_only(y, sr, n_mfcc, n_mels, hop_length, n_fft):
    # MFCC + Δ + Δ² ONLY (no extra spectral features)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_mels=n_mels,
                                hop_length=hop_length, n_fft=n_fft)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    def agg_rows(M):
        return np.concatenate([stats_1d(M[i, :]) for i in range(M.shape[0])], axis=0)

    feat = np.concatenate([agg_rows(mfcc), agg_rows(d1), agg_rows(d2)], axis=0)
    return feat.astype(np.float32)

def extract_features_file(path, cfg):
    y, sr = librosa.load(path, sr=cfg["sr"], mono=True)
    if y.size == 0:
        raise RuntimeError(f"Empty or unreadable audio: {path}")
    if cfg.get("trim_top_db") is not None:
        y, _ = librosa.effects.trim(y, top_db=cfg["trim_top_db"])
        if y.size == 0:
            raise RuntimeError(f"Audio became empty after trim: {path}")

    frames = frame_audio(y, sr, cfg.get("win_sec", 2.0), cfg.get("hop_sec", 1.0))
    feats = [extract_features_window_mfcc_only(f, sr, cfg["n_mfcc"], cfg["n_mels"],
                                               cfg["hop_length"], cfg["n_fft"]) for f in frames]
    return np.stack(feats)

def main():
    if len(sys.argv) < 2:
        print("Usage: python rf_mfcc_infer.py <audio_file> [model_path]")
        sys.exit(1)

    model_path = sys.argv[2] if len(sys.argv) >= 3 else MODEL_PATH
    payload = load(model_path)
    model = payload["model"]
    class_names = payload["class_names"]
    cfg = payload["feature_config"]

    X = extract_features_file(sys.argv[1], cfg)

    # Safety: ensure feature dimension matches the trained pipeline
    try:
        expected = model.named_steps["scaler"].n_features_in_
        if X.shape[1] != expected:
            print(f"[ERROR] Feature size mismatch: got {X.shape[1]}, expected {expected}. "
                  f"Make sure this script uses MFCC-only to match training.")
            sys.exit(2)
    except Exception:
        pass

    y_pred = model.predict(X)
    counts = Counter(y_pred.tolist())
    winner = counts.most_common(1)[0][0]
    print(f"Predicted class: {class_names[int(winner)]}")
    votes = {class_names[int(k)]: v for k, v in counts.items()}
    print(f"Votes per class: {votes}")

if __name__ == "__main__":  
    main()
