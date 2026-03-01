# eval.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

from models.vit import VIT
from dataloaders.dataset import create_dataloader

def evaluate(model, dataloader, device, class_names):
    model.eval()
    y_true = []
    y_pred = []

    print("[INFO] Đang chạy dự đoán trên tập Test...")
    with torch.inference_mode():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            
            predictions = torch.argmax(outputs, dim=1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png"):

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('Thực tế (True Label)', fontsize=12)
    plt.xlabel('Dự đoán (Predicted Label)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300) # Lưu với độ phân giải cao chuẩn bài báo
    print(f"\n[INFO] Đã lưu biểu đồ Confusion Matrix tại: {save_path}")

def main():
    print("\n" + "="*60)
    print("📊 BẮT ĐẦU ĐÁNH GIÁ MÔ HÌNH (EVALUATION)")
    print("="*60)

    DATA_DIR = "data"
    BATCH_SIZE = 16
    IMAGE_SIZE = 224
    WEIGHTS_PATH = "weights/optimized_vit_best.pth" # Đường dẫn file weights sau khi train
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    _, test_loader, class_names = create_dataloader(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_worker=4
    )
    num_classes = len(class_names)

    print("\n[INFO] Đang khởi tạo OptimizedViT...")
    model = VIT(num_classes=num_classes, num_layers=6).to(device)
    
    if os.path.exists(WEIGHTS_PATH):
        print(f"[INFO] Đang load trọng số từ: {WEIGHTS_PATH}")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    else:
        print(f"❌ Lỗi: Không tìm thấy file trọng số tại {WEIGHTS_PATH}.")
        print("💡 Vui lòng chạy file train.py trước để huấn luyện mô hình!")
        return

    # 4. Chạy đánh giá
    y_true, y_pred = evaluate(model, test_loader, device, class_names)

    # 5. Tính toán các Metrics chuẩn bài báo
    acc = accuracy_score(y_true, y_pred)
    # Dùng macro average để đánh giá công bằng cho mọi class (kể cả class ít ảnh)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print("\n" + "="*60)
    print("🏆 KẾT QUẢ ĐÁNH GIÁ TỔNG THỂ (OVERALL METRICS)")
    print("="*60)
    print(f"▪️ Accuracy  : {acc * 100:.2f}%")
    print(f"▪️ Precision : {precision * 100:.2f}%")
    print(f"▪️ Recall    : {recall * 100:.2f}%")
    print(f"▪️ F1-Score  : {f1 * 100:.2f}%")
    
    print("\n" + "="*60)
    print("📋 BÁO CÁO CHI TIẾT TỪNG LỚP (CLASSIFICATION REPORT)")
    print("="*60)
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    # 6. Vẽ Confusion Matrix
    plot_confusion_matrix(y_true, y_pred, class_names)

if __name__ == "__main__":
    main()