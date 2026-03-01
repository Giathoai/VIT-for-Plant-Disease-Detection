import torch
from models.vit import VIT
from dataloaders.dataset import create_dataloader
from utils import engine
from utils.helpers import set_seeds

def main():
    DATA_DIR = "data"
    BATCH_SIZE = 16
    IMAGE_SIZE = 224
    EPOCHS = 10  
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")

    train_dataloader, test_dataloader, class_names = create_dataloader(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        num_worker=4
    )

    vit = VIT(num_classes=len(class_names)).to(device)

    optimizer = torch.optim.AdamW(params=vit.parameters(), lr=0.0001)
    
    loss_fn = torch.nn.CrossEntropyLoss()

    set_seeds(42)

    results = engine.train(model=vit,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=EPOCHS,
                           device=device)
    
    torch.save(vit.state_dict(), "weights/optimized_vit_best.pth")
    print("\n[INFO] Đã lưu mô hình tại weights/optimized_vit_best.pth")

if __name__ == "__main__":
    main()