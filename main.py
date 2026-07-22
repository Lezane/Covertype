import os
import copy
import argparse
import numpy as np

# Headless matplotlib backend to avoid X11 crashes on SSH clusters
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, TensorDataset

# Only import CIFAR dependencies if requested to save cluster memory
try:
    from torchvision.models import vgg11, vgg11_bn, vgg19_bn, resnet18, resnet50, mobilenet_v2
    from datasets import load_dataset
except ImportError:
    pass

# =========================================================
# 1. SHARED OPTIMIZER: MUON
# =========================================================
class Muon(Optimizer):
    """ Muon - Momentum Orthogonalized by Newton-Schulz """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr, momentum, nesterov = group['lr'], group['momentum'], group['nesterov']
            for p in group['params']:
                if p.grad is None or p.grad.ndim == 0:
                    continue
                g = p.grad
                state = self.state[p]
                
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                g = g.add(buf, alpha=momentum) if nesterov else buf.clone()
                
                if g.ndim >= 2:
                    orig_shape = g.shape
                    g = g.view(orig_shape[0], -1)
                    g = g / (g.norm() + 1e-8)
                    a, b, c = (3.4445, -4.7750, 2.0315)
                    for _ in range(5):
                        if g.size(0) < g.size(1):
                            A = g @ g.T
                            g = a * g + b * A @ g + c * A @ A @ g
                        else:
                            A = g.T @ g
                            g = a * g + b * g @ A + c * g @ A @ A
                    g = g.view(orig_shape)
                p.add_(g, alpha=-lr)
        return loss

# =========================================================
# 2. CIFAR-10 SPECIFIC LOGIC (Computer Vision)
# =========================================================
def get_cifar10_model(model_name, device):
    num_classes = 9
    if model_name == 'mobilenet_v2':
        model = mobilenet_v2(weights=None, num_classes=num_classes)
    elif model_name.startswith('vgg'):
        model_fns = {'vgg11': vgg11, 'vgg11_bn': vgg11_bn, 'vgg19_bn': vgg19_bn}
        model = model_fns[model_name](weights=None, num_classes=num_classes)
        model.avgpool = nn.Identity()
        model.classifier = nn.Linear(512, num_classes)
    elif model_name.startswith('resnet'):
        model_fns = {'resnet18': resnet18, 'resnet50': resnet50}
        model = model_fns[model_name](weights=None, num_classes=num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    else:
        raise ValueError(f"Model {model_name} not supported.")

    # Initialize weights
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or (isinstance(m, nn.Linear) and m.out_features != num_classes):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear) and m.out_features == num_classes:
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    for m in model.modules():
        if type(m).__name__ in ['Bottleneck', 'BasicBlock']:
            nn.init.constant_(m.bn3.weight if hasattr(m, 'bn3') else m.bn2.weight, 0)

    return model.to(device)

def get_cifar10_dataloaders(device, batch_size, remove_percentage):
    hf_dataset = load_dataset("uoft-cs/cifar10")
    
    def create_dataset(train=True):
        split = 'train' if train else 'test'
        raw_set = hf_dataset[split]
        targets = np.array(raw_set['label'])
        data = np.stack([np.array(img) for img in raw_set['img']])

        plane_idx, car_idx, other_idx = np.where(targets == 0)[0], np.where(targets == 1)[0], np.where(targets > 1)[0]
        rng = np.random.RandomState(42)

        # Calculate dynamic keep ratio based on passed argument
        keep_cars_ratio = (100.0 - remove_percentage) / 100.0
        
        if train:
            keep_planes = int(len(plane_idx) * 0.95)
            keep_cars = int(len(car_idx) * keep_cars_ratio)
        else:
            # For testing, we keep 50% to maintain a balanced test set comparison
            keep_planes = int(len(plane_idx) * 0.50)
            keep_cars = int(len(car_idx) * 0.50)
            
        indices = np.concatenate([rng.choice(plane_idx, keep_planes, False), rng.choice(car_idx, keep_cars, False), other_idx])
        rng.shuffle(indices)

        orig_labels = targets[indices]
        new_labels = np.where(orig_labels <= 1, 0, orig_labels - 1)
        is_original_car = np.where(orig_labels == 1, 1, 0)

        X = torch.tensor(data[indices]).permute(0, 3, 1, 2).float().to(device) / 255.0
        Y = torch.tensor(new_labels).long().to(device)
        is_car = torch.tensor(is_original_car).long().to(device)

        mean, std = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1).to(device), torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1).to(device)
        return TensorDataset((X - mean) / std, Y, is_car)

    train_ds = create_dataset(train=True)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0),
            DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0),
            DataLoader(create_dataset(train=False), batch_size=batch_size, shuffle=False, num_workers=0))


# =========================================================
# 3. UNIFIED TRAINING LOOP (Smart Tracking for Both)
# =========================================================
def train_and_track(model, optimizers, trainloader, evalloader, testloader, device, epochs, exp_name, output_dir, is_cifar=False):
    criterion = nn.CrossEntropyLoss()
    if not isinstance(optimizers, list): optimizers = [optimizers]
    metrics = {'train_acc_s0': [], 'train_acc_s1': [], 'test_acc_s0': [], 'test_acc_s1': []}

    def get_acc(m, loader):
        m.eval()
        correct_s0, total_s0, correct_s1, total_s1 = 0, 0, 0, 0
        with torch.no_grad():
            for batch in loader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                _, predicted = torch.max(m(inputs), 1)

                # DYNAMIC CATCH: If DataLoader returns 3 items = CIFAR10. If 2 items = Covertype.
                if is_cifar and len(batch) > 2:
                    is_orig_car = batch[2].to(device)
                    mask_s0 = (is_orig_car == 1)
                    mask_s1 = (is_orig_car == 0)
                else:
                    # For Covertype (standard Tabular Dataset format)
                    mask_s0 = torch.ones_like(labels, dtype=torch.bool)
                    mask_s1 = torch.zeros_like(labels, dtype=torch.bool)

                total_s0 += mask_s0.sum().item()
                total_s1 += mask_s1.sum().item()
                correct_s0 += (predicted[mask_s0] == labels[mask_s0]).sum().item()
                correct_s1 += (predicted[mask_s1] == labels[mask_s1]).sum().item()

        return (100 * correct_s0 / total_s0 if total_s0 > 0 else 0.0), (100 * correct_s1 / total_s1 if total_s1 > 0 else 0.0)

    for epoch in range(epochs + 1):
        if epoch > 0:
            model.train()
            for batch in trainloader:
                inputs, labels = batch[0].to(device), batch[1].to(device)
                for opt in optimizers: opt.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                for opt in optimizers: opt.step()

        train_s0, train_s1 = get_acc(model, evalloader)
        test_s0, test_s1 = get_acc(model, testloader)

        metrics['train_acc_s0'].append(train_s0); metrics['train_acc_s1'].append(train_s1)
        metrics['test_acc_s0'].append(test_s0); metrics['test_acc_s1'].append(test_s1)

        if is_cifar:
            print(f"[{exp_name}] Epoch {epoch:03d}/{epochs} | Train S0: {train_s0:5.2f}% | Train S1: {train_s1:5.2f}% || Test S0: {test_s0:5.2f}% | Test S1: {test_s1:5.2f}%")
        else:
            print(f"[{exp_name}] Epoch {epoch:03d}/{epochs} | Train Acc: {train_s0:5.2f}% || Test Acc: {test_s0:5.2f}%")

    # Plot creation (Headless save to disk)
    os.makedirs(output_dir, exist_ok=True)
    if is_cifar:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{exp_name} (CIFAR-10)", fontsize=16)
        axs[0,0].plot(metrics['train_acc_s0'], color='tab:blue'); axs[0,0].set_title(f"Train S0: {train_s0:.2f}%")
        axs[0,1].plot(metrics['train_acc_s1'], color='tab:orange'); axs[0,1].set_title(f"Train S1: {train_s1:.2f}%")
        axs[1,0].plot(metrics['test_acc_s0'], color='tab:green'); axs[1,0].set_title(f"Test S0: {test_s0:.2f}%")
        axs[1,1].plot(metrics['test_acc_s1'], color='tab:red'); axs[1,1].set_title(f"Test S1: {test_s1:.2f}%")
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(metrics['train_acc_s0'], label='Train Accuracy')
        ax.plot(metrics['test_acc_s0'], label='Test Accuracy')
        ax.set_title(f"{exp_name} (Covertype)", fontsize=14, fontweight='bold')
        ax.set_xlabel('Epochs'); ax.set_ylabel('Accuracy (%)')
        ax.legend()
        
    for ax in axs.flat if is_cifar else [ax]:
        ax.grid(True); ax.set_ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{exp_name.replace(' ', '_').lower()}.png"))
    plt.close(fig)

    return metrics

# =========================================================
# 4. MAIN ENTRY POINT & ROUTER
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="Unified Training: Covertype & CIFAR-10")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'covertype'], help="Dataset to run")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--arch', type=str, default='vgg19_bn', help="Architecture for CIFAR10")
    
    # >>> NEW ARGUMENT: dynamic remove_percentage
    parser.add_argument('--remove-percentage', type=float, default=95.0, help="Percentage of CIFAR-10 Cars to remove in training")
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=str, default='./plots')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    if args.dataset == 'cifar10':
        print(f"Using CIFAR-10 with remove-percentage: {args.remove_percentage}%")
        train_ldr, eval_ldr, test_ldr = get_cifar10_dataloaders(device, args.batch_size, args.remove_percentage)
        base_model = get_cifar10_model(args.arch, device)
    else:
        # >>> INTEGRATION POINT FOR EXISTING COVERTYPE FILES <<<
        # Import your existing Covertype model and dataloaders here.
        # Example (assuming your files are named `data_loader.py` and `model.py`):
        # 
        # from data_loader import get_covertype_loaders
        # from model import CovertypeNet
        # train_ldr, eval_ldr, test_ldr = get_covertype_loaders(device, args.batch_size)
        # base_model = CovertypeNet().to(device)
        
        print("Integration Note: Please uncomment and import your Covertype dataloaders and model at line ~242 of main.py!")
        return

    initial_state = copy.deepcopy(base_model.state_dict())
    is_cf = (args.dataset == 'cifar10')

    # --- Run 1: AdamW ---
    print(f"\n=== Training AdamW on {args.dataset.upper()} ===")
    model_adamw = copy.deepcopy(base_model)
    model_adamw.load_state_dict(initial_state)
    opt_adamw = optim.AdamW(model_adamw.parameters(), lr=1e-3, weight_decay=1e-2)
    train_and_track(model_adamw, opt_adamw, train_ldr, eval_ldr, test_ldr, device, args.epochs, f"AdamW_{args.dataset}_{args.remove_percentage}pct", args.output_dir, is_cifar=is_cf)

    # --- Run 2: SGD ---
    print(f"\n=== Training SGD on {args.dataset.upper()} ===")
    model_sgd = copy.deepcopy(base_model)
    model_sgd.load_state_dict(initial_state)
    opt_sgd = optim.SGD(model_sgd.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
    train_and_track(model_sgd, opt_sgd, train_ldr, eval_ldr, test_ldr, device, args.epochs, f"SGD_{args.dataset}_{args.remove_percentage}pct", args.output_dir, is_cifar=is_cf)

    # --- Run 3: Muon ---
    print(f"\n=== Training Muon on {args.dataset.upper()} ===")
    model_muon = copy.deepcopy(base_model)
    model_muon.load_state_dict(initial_state)
    
    muon_params = [p for p in model_muon.parameters() if p.ndim >= 2]
    other_params = [p for p in model_muon.parameters() if p.ndim < 2]
    opt_muon = Muon(muon_params, lr=0.02, momentum=0.95, nesterov=True)
    opt_other = optim.AdamW(other_params, lr=1e-3, weight_decay=1e-2)
    train_and_track(model_muon, [opt_muon, opt_other], train_ldr, eval_ldr, test_ldr, device, args.epochs, f"Muon_{args.dataset}_{args.remove_percentage}pct", args.output_dir, is_cifar=is_cf)

if __name__ == '__main__':
    main()
