import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from tqdm import tqdm
from PIL import Image

# ==========================================
# 1. DATASET & PREPROCESSING
# ==========================================
class WBCDataset(Dataset):
    def __init__(self, df, folder, transform=None, is_test=False):
        self.df = df
        self.folder = folder
        self.transform = transform
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]['ID']
        img_path = os.path.join(self.folder, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        if self.is_test:
            return image, img_name
        
        return image, self.df.iloc[idx]['label_encoded']

def get_transforms(img_size):
    # Augmentations spécifiques à la biologie (rotations totales)
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90), 
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return train_tf, val_tf

# ==========================================
# 2. MODÈLE (ResNet18 Transfer Learning)
# ==========================================
class ResNetWBC(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Chargement du ResNet pré-entraîné
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Modification de la dernière couche
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

# ==========================================
# 3. BOUCLE D'ENTRAÎNEMENT
# ==========================================
def train():
    # Config
    DATA_PATH = '../IMA205-challenge'
    IMG_SIZE = 224 # ResNet préfère 224x224
    BATCH_SIZE = 32
    EPOCHS = 15
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Chargement Meta
    train_meta = pd.read_csv(f'{DATA_PATH}/train_metadata.csv')
    le = LabelEncoder()
    train_meta['label_encoded'] = le.fit_transform(train_meta['label'])
    
    # Gestion du déséquilibre des classes (Weighted Loss)
    counts = train_meta['label_encoded'].value_counts().sort_index().values
    weights = 1.0 / torch.tensor(counts, dtype=torch.float)
    weights = weights.to(DEVICE)

    # Split
    train_df, val_df = train_test_split(
        train_meta, test_size=0.15, stratify=train_meta['label_encoded'], random_state=42
    )

    t_tf, v_tf = get_transforms(IMG_SIZE)
    train_loader = DataLoader(WBCDataset(train_df, f'{DATA_PATH}/train', t_tf), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WBCDataset(val_df, f'{DATA_PATH}/train', v_tf), batch_size=BATCH_SIZE)

    # Init Modèle & Optim
    model = ResNetWBC(len(le.classes_)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # LR plus faible pour le fine-tuning
    criterion = nn.CrossEntropyLoss(weight=weights)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)

    best_f1 = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                out = model(imgs.to(DEVICE))
                all_preds.extend(out.argmax(1).cpu().numpy())
                all_labels.extend(labels.numpy())
        
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Loss: {total_loss/len(train_loader):.4f} | F1: {val_f1:.4f}")
        
        # Step scheduler
        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_resnet.pth')

    print(f"Entraînement fini. Meilleur F1: {best_f1:.4f}")

if __name__ == "__main__":
    train()