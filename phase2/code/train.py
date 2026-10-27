import torch
from torch.utils.data import DataLoader
from dataset import RGBThermalDataset
from train_model import FusionYOLO

# -----------------------
# CONFIG
# -----------------------
DEVICE = "cpu"          # macOS CPU
BATCH_SIZE = 4
EPOCHS = 1
LR = 5e-5
CHECKPOINT_EVERY = 50

print("✅ train.py started")

# -----------------------
# DATASET
# -----------------------
print("📦 Loading dataset...")
train_dataset = RGBThermalDataset(
    root="phase2/dataset",
    split="train"
)

print(f"📊 Dataset length: {len(train_dataset)}")

def collate_fn(batch):
    rgbs, irs, targets = zip(*batch)
    rgbs = torch.stack(rgbs)
    irs = torch.stack(irs)
    return rgbs, irs, list(targets)  # variable-length labels

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

print(f"📦 Dataloader length: {len(train_loader)}")
print(f"🔁 Epochs: {EPOCHS}")

# -----------------------
# MODEL
# -----------------------
print("🧠 Initializing model...")
model = FusionYOLO().to(DEVICE)

# 🔒 Freeze pretrained YOLO backbones
for p in model.rgb_backbone.parameters():
    p.requires_grad = False

for p in model.ir_backbone.parameters():
    p.requires_grad = False

# Train fusion layers only
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

print("🚀 Training setup complete")
print("🚀 Training started\n")

# -----------------------
# TRAIN LOOP
# -----------------------
for epoch in range(EPOCHS):
    print(f"📘 Epoch {epoch + 1}/{EPOCHS}")
    model.train()

    for i, (rgbs, irs, targets) in enumerate(train_loader):
        rgbs = rgbs.to(DEVICE)
        irs = irs.to(DEVICE)

        optimizer.zero_grad()

        _, loss, _ = model(rgbs, irs, targets)

        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"  Batch {i}/{len(train_loader)} | Loss: {loss.item():.6f}")

        if i % CHECKPOINT_EVERY == 0 and i > 0:
            torch.save(
                model.state_dict(),
                f"phase2_fusion_epoch{epoch}_batch{i}.pt"
            )

    torch.save(
        model.state_dict(),
        f"phase2_fusion_epoch{epoch}_final.pt"
    )

print("\n🎉 Phase-2 training complete")
