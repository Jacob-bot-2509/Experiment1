import argparse
import yaml
import torch
from data_loader import get_dataloaders
from model import get_model
from utils import set_seed
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main(config_path):
    import os, sys
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

    _, _, test_loader = get_dataloaders(
        config['data']['data_dir'],
        config['data']['batch_size'],
        config['data']['valid_split'],
        config['data']['num_workers']
    )

    model = get_model(config['model']['name'], num_classes=10).to(device)
    model_path = os.path.join(config['output']['run_dir'], config['output']['save_model'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {acc:.4f}")
    print("\\\\nClassification Report:\\\\n", classification_report(all_labels, all_preds, target_names=[
        'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['airplane','auto','bird','cat','deer','dog','frog','horse','ship','truck'],
                yticklabels=['airplane','auto','bird','cat','deer','dog','frog','horse','ship','truck'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(config['output']['run_dir'], "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"Confusion matrix saved to {cm_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    main(args.config)