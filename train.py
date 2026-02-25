import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from tqdm import tqdm  # Import de la barre de progression

from dataset import WBCDataset, get_transforms
from cnn import SimpleCNN

# Config
DATA_PATH = '../IMA205-challenge'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 10
DEVICE  = torch.device("cpu")
#torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

def train():
    # 1. Chargement et Encodage
    train_meta = pd.read_csv(f'{DATA_PATH}/train_metadata.csv')
    test_meta = pd.read_csv(f'{DATA_PATH}/test_metadata.csv')
    
    le = LabelEncoder()
    train_meta['label_encoded'] = le.fit_transform(train_meta['label'])
    
    train_df, val_df = train_test_split(
        train_meta, test_size=0.2, stratify=train_meta['label_encoded'], random_state=42
    )

    t_tf, v_tf = get_transforms(IMG_SIZE)
    
    train_loader = DataLoader(WBCDataset(train_df, f'{DATA_PATH}/train', t_tf), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(WBCDataset(val_df, f'{DATA_PATH}/train', v_tf), batch_size=BATCH_SIZE)
    test_loader = DataLoader(WBCDataset(test_meta, f'{DATA_PATH}/test', v_tf, is_test=True), batch_size=BATCH_SIZE)

    # 2. Setup Modèle
    model = SimpleCNN(len(le.classes_), IMG_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 3. Boucle d'entraînement
    best_f1 = 0
    print(f"Début de l'entraînement sur {DEVICE}...")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        # Ajout de tqdm ici
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # Eval
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Eval]"):
                out = model(imgs.to(DEVICE))
                preds.extend(out.argmax(1).cpu().numpy())
                targets.extend(labels.numpy())
        
        score = f1_score(targets, preds, average='macro')
        print(f"Summary -> Loss: {train_loss/len(train_loader):.4f} | F1: {score:.4f}")
        
        if score > best_f1:
            best_f1 = score
            torch.save(model.state_dict(), 'best_model.pth')
            print("  --> Modèle sauvegardé !")

    # 4. Génération de la Soumission
    print("\nGénération du fichier submission.csv...")
    if torch.cuda.is_available() or hasattr(torch.backends, 'mps'):
        model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
    model.eval()
    
    ids, final_preds = [], []
    with torch.no_grad():
        for imgs, img_ids in tqdm(test_loader, desc="Inference"):
            out = model(imgs.to(DEVICE))
            final_preds.extend(out.argmax(1).cpu().numpy())
            ids.extend(img_ids)
    
    submission_df = pd.DataFrame({
        'ID': ids, 
        'label': le.inverse_transform(final_preds)
    })
    submission_df.to_csv('submission.csv', index=False)
    print(f"Terminé ! Score F1 max atteint : {best_f1:.4f}")

if __name__ == "__main__":
    train()