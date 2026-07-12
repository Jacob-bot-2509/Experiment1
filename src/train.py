import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from data_loader import get_dataloaders
from model import get_model
from utils import set_seed, plot_curves
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def main(config_path):
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, config_path)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['train']['seed'])
    config['output']['run_dir'] = os.path.normpath(
        os.path.join(project_root, config['output']['run_dir'])
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader = get_dataloaders(
        config['data']['data_dir'],
        config['data']['batch_size'],
        config['data']['valid_split'],
        config['data']['num_workers']
    )

    model = get_model(config['model']['name'], num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=float(config['train']['learning_rate']),
                          momentum=float(config['train']['momentum']),
                          weight_decay=float(config['train']['weight_decay']))
    scheduler = StepLR(optimizer, step_size=int(config['train']['lr_step_size']), gamma=0.1)

    os.makedirs(os.path.join(project_root, config['output']['run_dir']), exist_ok=True)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0

    for epoch in range(1, int(config['train']['epochs']) + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, valid_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch {epoch}/{int(config['train']['epochs'])} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       os.path.join(project_root, config['output']['run_dir'], config['output']['save_model']))
            print(f"  -> Saved best model with val_acc: {val_acc:.4f}")

    plot_curves(train_losses, val_losses, train_accs, val_accs,
                os.path.join(project_root, config['output']['run_dir'], "learning_curves.png"))
    print("Training finished. Best validation accuracy: {:.4f}".format(best_val_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)