import os
import glob
import math
import random
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import numpy as np
import librosa

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from joblib import dump

# =========================
DATA_DIR   = "data"
TEST_DIR = "data_test"
SR          = 16000
N_MFCC      = 40
N_MELS      = 64
HOP_LENGTH  = 512
N_FFT       = 1024

# Windowing
WIN_SEC     = 2.0
HOP_SEC     = 0.25

# Clean-up
TRIM_TOP_DB = 10
PRE_EMPHASIS = 0.97

# Augmentation (training only)
USE_AUG                = True
AUG_TARGET_MULTIPLIER  = 1.2
AUG_NOISE_STD          = (0.005, 0.02)
AUG_SHIFT_FRAC         = 0.10
AUG_STRETCH            = (0.9, 1.1)
AUG_GAIN               = (0.8, 1.2)

# Model / search
RANDOM_STATE = 42
MODEL_OUT    = "infant_cry_svm_mfcc_pca_svm.joblib"
PARAM_GRID = {
    "pca__n_components": [48, 64, 96, 128],
    "rf__n_estimators": [200, 400, 600],
    "rf__max_depth": [None, 20, 40, 60],
    "rf__min_samples_split": [2, 5, 10],
}

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
rng = np.random.default_rng(RANDOM_STATE)
random.seed(RANDOM_STATE)


# Audio / feature utils
def list_class_files(root: str) -> Tuple[List[str], List[str], List[str]]:
    classes = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    if not classes:
        raise RuntimeError(f"No class folders found in {root}")
    paths, labels = [], []
    for cls in classes:
        folder = os.path.join(root, cls)
        files = []
        for ext in AUDIO_EXTS:
            files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        for f in files:
            paths.append(f)
            labels.append(cls)
    if not paths:
        raise RuntimeError(f"No audio files found under {root}")
    return paths, labels, classes

def pre_emphasize(y: np.ndarray, coeff: float) -> np.ndarray:
    if not coeff:
        return y.astype(np.float32)
    y = y.astype(np.float32)
    return np.concatenate(([y[0]], y[1:] - coeff * y[:-1]))

def frame_audio(y: np.ndarray, sr: int, win_sec: float, hop_sec: float) -> List[np.ndarray]:
    win = int(round(win_sec * sr)); hop = int(round(hop_sec * sr))
    if len(y) < win:
        out = np.zeros(win, dtype=y.dtype); out[:len(y)] = y
        return [out]
    return [y[i:i+win] for i in range(0, len(y) - win + 1, hop)]

def stats_1d(v: np.ndarray) -> np.ndarray:
    p10 = np.percentile(v, 10); p90 = np.percentile(v, 90)
    return np.array([np.mean(v), np.std(v), np.median(v), p10, p90, p90 - p10], dtype=np.float32)

def features_mfcc(y: np.ndarray, sr: int) -> np.ndarray:
    """
    MFCC-only: MFCC + Δ + Δ²; each coefficient summarized with mean/std/median/p10/p90/spread
    shape = N_MFCC * 3 * 6 = 760 when N_MFCC = 40
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_mels=N_MELS,
                                hop_length=HOP_LENGTH, n_fft=N_FFT)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    def agg_rows(M):
        return np.concatenate([stats_1d(M[i, :]) for i in range(M.shape[0])], axis=0)

    feat = np.concatenate([agg_rows(mfcc), agg_rows(d1), agg_rows(d2)], axis=0)
    return feat.astype(np.float32)

def augment_window(w: np.ndarray) -> np.ndarray:
    x = w.astype(np.float32).copy()
    ops = []
    if AUG_GAIN:         ops.append("gain")
    if AUG_SHIFT_FRAC>0: ops.append("shift")
    if AUG_STRETCH:      ops.append("stretch")
    if AUG_NOISE_STD:    ops.append("noise")
    rng.shuffle(ops)
    n_apply = rng.integers(1, min(2, len(ops)) + 1)

    for op in ops[:n_apply]:
        if op == "gain":
            g = float(rng.uniform(AUG_GAIN[0], AUG_GAIN[1])); x = x * g
        elif op == "shift":
            max_shift = int(len(x) * AUG_SHIFT_FRAC)
            s = int(rng.integers(-max_shift, max_shift + 1)); x = np.roll(x, s)
        elif op == "stretch":
            rate = float(rng.uniform(AUG_STRETCH[0], AUG_STRETCH[1]))
            try:
                xs = librosa.effects.time_stretch(x, rate=rate)
            except Exception:
                xs = x
            if len(xs) >= len(x): x = xs[:len(x)]
            else: out = np.zeros_like(x); out[:len(xs)] = xs; x = out
        elif op == "noise":
            std = float(rng.uniform(AUG_NOISE_STD[0], AUG_NOISE_STD[1])) * (np.std(x) + 1e-9)
            noise = rng.normal(0.0, std, size=len(x)).astype(np.float32); x = x + noise
    return x

def build_window_dataset(file_paths: List[str],
                         file_labels_idx: np.ndarray,
                         augment: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      X: (n_windows, n_feat)
      y: (n_windows,)
      groups: (n_windows,)  # file index for GroupKFold / file-level voting
    """
    X_list, y_list, g_list = [], [], []
    raw_windows_by_class: Dict[int, List[np.ndarray]] = defaultdict(list)

    for file_idx, (p, li) in enumerate(zip(file_paths, file_labels_idx)):
        y, sr = librosa.load(p, sr=SR, mono=True)
        if y.size == 0:
            continue
        y = pre_emphasize(y, PRE_EMPHASIS)
        if TRIM_TOP_DB is not None:
            y, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
            if y.size == 0:
                continue
        frames = frame_audio(y, sr, WIN_SEC, HOP_SEC)
        for w in frames:
            raw_windows_by_class[int(li)].append(w)
            X_list.append(features_mfcc(w, sr))
            y_list.append(int(li))
            g_list.append(file_idx)

    if not X_list:
        raise RuntimeError("No windows produced – check audio files and config.")

    if augment and USE_AUG and AUG_TARGET_MULTIPLIER > 0:
        counts = Counter(y_list)
        maj = max(counts.values())
        target = int(math.ceil(maj * AUG_TARGET_MULTIPLIER))
        next_gid = max(g_list) + 1 if g_list else 0
        for c, pool in raw_windows_by_class.items():
            n = counts.get(c, 0)
            need = target - n
            if need <= 0 or not pool:
                continue
            for _ in range(need):
                base = random.choice(pool)
                augw = augment_window(base)
                X_list.append(features_mfcc(augw, SR))
                y_list.append(c)
                g_list.append(next_gid)
                next_gid += 1

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    groups = np.array(g_list, dtype=np.int64)
    return X, y, groups


# Train / Optional external test
def main():
    # -------- TRAIN on TRAIN_DIR --------
    print("[INFO] Loading TRAIN files…")
    train_paths, train_labels_str, class_names = list_class_files(DATA_DIR)
    cls_to_idx = {c: i for i, c in enumerate(class_names)}
    y_file_train = np.array([cls_to_idx[s] for s in train_labels_str], dtype=np.int64)

    print("[INFO] Building TRAIN windows…")
    X_train, y_train, g_train = build_window_dataset(train_paths, y_file_train, augment=True)

    # Show distribution
    train_counts = Counter(y_train)
    human_train = {class_names[k]: int(train_counts[k]) for k in sorted(train_counts)}
    print(f"[INFO] Classes: {class_names}")
    print(f"[INFO] Train windows: {len(y_train)} | per-class: {human_train}")

    # Pipeline
    pipe = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("pca", PCA(random_state=RANDOM_STATE)),
    ("rf", RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )),
])


    # CV on TRAIN only (group by file)
    gkf = GroupKFold(n_splits=5)
    search = GridSearchCV(
        pipe,
        param_grid=PARAM_GRID,
        cv=gkf,
        scoring="balanced_accuracy",   
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    print("[INFO] Training (GroupKFold grid search by file)…")
    search.fit(X_train, y_train, groups=g_train)
    print(f"[INFO] Best params: {search.best_params_}")
    print(f"[INFO] CV best balanced_acc: {search.best_score_:.4f}")

    best_model = search.best_estimator_

    # Save model payload
    payload = {
        "model": best_model,
        "class_names": class_names,
        "feature_config": {
            "sr": SR, "n_mfcc": N_MFCC, "n_mels": N_MELS,
            "hop_length": HOP_LENGTH, "n_fft": N_FFT,
            "win_sec": WIN_SEC, "hop_sec": HOP_SEC,
            "trim_top_db": TRIM_TOP_DB
        }
    }
    dump(payload, MODEL_OUT)
    print(f"\n[INFO] Saved model to: {MODEL_OUT}")

    # -------- OPTIONAL: Evaluate on external TEST_DIR --------
    if TEST_DIR:
        print(f"\n[INFO] Evaluating on external test set: {TEST_DIR}")
        test_paths, test_labels_str, _ = list_class_files(TEST_DIR)

        # map test labels to train class indices; skip unknown classes
        keep_mask, y_file_test = [], []
        skipped = 0
        for lbl in test_labels_str:
            if lbl in cls_to_idx:
                keep_mask.append(True); y_file_test.append(cls_to_idx[lbl])
            else:
                keep_mask.append(False); skipped += 1
        test_paths = [p for p, keep in zip(test_paths, keep_mask) if keep]
        y_file_test = np.array(y_file_test, dtype=np.int64)
        if skipped:
            print(f"[WARN] Skipped {skipped} test files with unknown classes.")

        print("[INFO] Building TEST windows…")
        X_test, y_test_win, g_test = build_window_dataset(test_paths, y_file_test, augment=False)

        # Window-level metrics
        y_pred_win = best_model.predict(X_test)
        print("\n[WINDOW-LEVEL] Confusion Matrix:")
        print(confusion_matrix(y_test_win, y_pred_win))
        print("\n[WINDOW-LEVEL] Classification Report:")
        print(classification_report(y_test_win, y_pred_win, target_names=class_names, digits=4))
        print(f"[WINDOW-LEVEL] Accuracy: {accuracy_score(y_test_win, y_pred_win):.4f}")

        # File-level majority vote (true label from y_file_test[gid])
        y_proba_win = best_model.predict_proba(X_test)

        file_probs = defaultdict(list)
        for proba, gid in zip(y_proba_win, g_test):
            file_probs[int(gid)].append(proba)

        y_true_file, y_pred_file = [], []
        for gid, prob_list in file_probs.items():
            true_label = int(y_file_test[int(gid)])
            mean_proba = np.mean(prob_list, axis=0)   # average over windows
            winner = int(np.argmax(mean_proba))
            y_true_file.append(true_label)
            y_pred_file.append(winner)

        print("\n[FILE-LEVEL] Confusion Matrix:")
        print(confusion_matrix(y_true_file, y_pred_file))
        print("\n[FILE-LEVEL] Classification Report:")
        print(classification_report(y_true_file, y_pred_file,
                                    target_names=class_names, digits=4))
        print(f"[FILE-LEVEL] Accuracy: {accuracy_score(y_true_file, y_pred_file):.4f}")

if __name__ == "__main__":
    main()
