import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

# Nguong chat luong cua lab nay la f1_score, KHONG phai accuracy.
# Ly do: bo du lieu Adult co ty le lop 75/25. Mot mo hinh doan bua
# "thu nhap thap" cho moi mau da dat accuracy 0.75 ma khong hoc duoc gi.
F1_THRESHOLD = 0.65


def train(
    params: dict,
    data_path: str = "data/train_batch1.csv",
    eval_path: str = "data/holdout.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho GradientBoostingClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia (holdout).

    Tra ve:
        f1 (float): diem F1 cua lop duong (thu nhap > 50K) tren tap holdout.
    """

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        tracking_uri = "sqlite:///mlflow.db"
        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    mlflow.set_tracking_uri(tracking_uri)

    if "sqlite" in tracking_uri and "MLFLOW_ARTIFACT_ROOT" not in os.environ:
        os.environ["MLFLOW_ARTIFACT_ROOT"] = "./mlartifacts"

    # TODO 1: Doc du lieu huan luyen va danh gia
    df_train = pd.read_csv(data_path)
    df_eval  = pd.read_csv(eval_path)

    # TODO 2: Tach dac trung (X) va nhan (y)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval  = df_eval.drop(columns=["target"])
    y_eval  = df_eval["target"]

    # Bonus 5: Kiem tra ty le lop duong va canh bao lech lac du lieu
    pos_ratio = float(y_train.mean())
    ref_ratio = 0.248
    ratio_diff = abs(pos_ratio - ref_ratio) * 100
    if ratio_diff > 5.0:
        print(f"[CANH BAO DATA DRIFT] Ty le lop duong ({pos_ratio:.1%}) lech > 5% so voi tham chieu ({ref_ratio:.1%})!")
    else:
        print(f"[DATA CHECK OK] Ty le lop duong: {pos_ratio:.1%} (tham chieu: {ref_ratio:.1%})")

    with mlflow.start_run():

        # TODO 3: Ghi nhan cac sieu tham so
        mlflow.log_params(params)

        # TODO 4: Khoi tao va huan luyen GradientBoostingClassifier
        # Goi y: su dung random_state=42 de dam bao tinh tai tao
        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        # TODO 5: Du doan tren tap holdout va tinh chi so
        # Chu y: f1_score o day tinh cho LOP DUONG (target = 1), khong dung average.
        preds = model.predict(X_eval)
        f1    = float(f1_score(y_eval, preds))
        acc   = float(accuracy_score(y_eval, preds))

        # Bonus 2: Quet nguong quyet dinh tu 0.1 den 0.9 de tim nguong toi uu
        import numpy as np
        probs = model.predict_proba(X_eval)[:, 1]
        best_threshold = 0.50
        best_f1 = f1
        for th in np.arange(0.1, 0.95, 0.05):
            th = round(float(th), 2)
            th_preds = (probs >= th).astype(int)
            th_f1 = float(f1_score(y_eval, th_preds))
            if th_f1 > best_f1:
                best_f1 = th_f1
                best_threshold = th

        # TODO 6: Ghi nhan chi so vao MLflow
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("best_threshold", best_threshold)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.log_metric("positive_class_ratio", pos_ratio)
        mlflow.sklearn.log_model(model, "model")

        # TODO 7: In ket qua ra man hinh
        print(f"F1: {f1:.4f} | Accuracy: {acc:.4f}")
        print(f"Bonus 2 - Nguong toi uu: {best_threshold:.2f} (F1 dat {best_f1:.4f} so voi {f1:.4f} o nguong 0.50)")

        # Bonus 3: Tao bao cao chi tiet precision / recall va confusion matrix
        from sklearn.metrics import classification_report, confusion_matrix
        cm = confusion_matrix(y_eval, preds)
        clf_rep = classification_report(y_eval, preds, target_names=["thu_nhap_thap", "thu_nhap_cao"])
        os.makedirs("outputs", exist_ok=True)
        detail_txt = f"""=== BÁO CÁO CHI TIẾT PRECISION / RECALL (BONUS 3) ===

1. MA TRẬN NHẦM LẪN (CONFUSION MATRIX):
   [TN, FP] -> [{cm[0][0]}, {cm[0][1]}]
   [FN, TP] -> [{cm[1][0]}, {cm[1][1]}]

2. BẢNG ĐÁNH GIÁ CHI TIẾT (PRECISION, RECALL, F1):
{clf_rep}
- F1-score (ngưỡng 0.50): {f1:.4f}
- Accuracy: {acc:.4f}

3. TỐI ƯU HÓA NGƯỠNG QUYẾT ĐỊNH (BONUS 2):
- Ngưỡng tốt nhất: {best_threshold:.2f} mang lại F1-score: {best_f1:.4f}

4. KIỂM SOÁT PHÂN PHỐI DỮ LIỆU (BONUS 5):
- Tỷ lệ lớp dương: {pos_ratio:.2%} (tham chiếu: {ref_ratio:.2%})
"""
        with open("outputs/detail.txt", "w", encoding="utf-8") as f:
            f.write(detail_txt)

        # TODO 8: Luu metrics ra file outputs/report.json
        # File nay duoc doc boi GitHub Actions o Buoc 2
        report_data = {
            "f1_score": f1,
            "accuracy": acc,
            "best_threshold": best_threshold,
            "best_f1_score": best_f1,
            "positive_class_ratio": pos_ratio,
        }
        with open("outputs/report.json", "w") as f:
            json.dump(report_data, f, indent=2)

        # TODO 9: Luu mo hinh ra file models/model.joblib
        # File nay duoc upload len cloud storage o Buoc 2
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.joblib")

    # TODO 10: Tra ve f1
    return f1


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
