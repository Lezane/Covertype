import os
import copy
import argparse
import random
import numpy as np

# Headless matplotlib backend to avoid display crashes on headless servers
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_covtype

# Optional dependencies loaded if they are present
try:
    from torchvision.models import vgg11, vgg11_bn, vgg19_bn, resnet18, resnet50, mobilenet_v2
    from datasets import load_dataset
except ImportError:
    pass

try:
    import rtdl
except ImportError:
    pass

# =========================================================
# 1. OPTIMIZER: MUON
# =========================================================
class Muon(optim.Optimizer):
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

                # Newton-Schulz orthogonalization for 2D+ tensors
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
# 2. NEURAL NETWORK ARCHITECTURES
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

    for m in model.modules():
        if isinstance(m, nn.Conv2d) or (isinstance(m, nn.Linear) and m.out_features != num_classes):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear) and m.out_features == num_classes:
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    for m in model.modules():
        if type(m).__name__ in ['Bottleneck', 'BasicBlock']:
            nn.init.constant_(m.bn3.weight if hasattr(m, 'bn3') else m.bn2.weight, 0)

    return model.to(device)

class FTTransformerWrapper(nn.Module):
    def __init__(self, num_features, cat_cardinalities):
        super().__init__()
        self.model = rtdl.FTTransformer.make_baseline(
            n_num_features=num_features, cat_cardinalities=cat_cardinalities,
            d_token=32, d_out=32, n_blocks=3, attention_dropout=0.2, ffn_d_hidden=64,
            ffn_dropout=0.1, residual_dropout=0.0, last_layer_query_idx=[-1]
        )
        self.head = nn.Linear(32, 2)

    def forward(self, x_num, x_cat):
        num_input = x_num if x_num.shape[1] > 0 else None
        cat_input = x_cat if x_cat.shape[1] > 0 else None
        x = self.model(num_input, cat_input)
        if x.dim() == 3: x = x.squeeze(1)
        return self.head(x)

class MLPWrapper(nn.Module):
    def __init__(self, num_features, cat_cardinalities, embed_dim=16, hidden_dims=[128, 64, 32]):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(card, embed_dim) for card in cat_cardinalities])
        in_dim = num_features + len(cat_cardinalities) * embed_dim
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], 2)

    def forward(self, x_num, x_cat):
        x_embeds = [self.embeddings[i](x_cat[:, i]) for i in range(x_cat.shape[1])] if x_cat is not None and x_cat.shape[1] > 0 else []
        x = x_num
        if x_embeds:
            x_cat_concat = torch.cat(x_embeds, dim=1)
            x = torch.cat([x, x_cat_concat], dim=1) if x is not None else x_cat_concat
        return self.head(self.mlp(x))

# =========================================================
# 3. DATALOADERS
# =========================================================
class TabularDataset(Dataset):
    def __init__(self, x_num, x_cat, y, s0):
        self.x_num, self.x_cat, self.y, self.s0 = x_num, x_cat, y, s0
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.x_num[idx], self.x_cat[idx], self.y[idx], self.s0[idx]

def get_covertype_dataloaders(batch_size=2048, random_seed=42):
    print("Downloading and preparing Covertype dataset...")
    covtype = fetch_covtype(as_frame=True)
    df = covtype.frame
    
    df['target_bin'] = (df['Cover_Type'] == 2).astype(int)
    
    wilderness_cols = [col for col in df.columns if col.startswith('Wilderness_Area')]
    soil_cols = [col for col in df.columns if col.startswith('Soil_Type')]
    
    df['Wilderness_Area'] = np.argmax(df[wilderness_cols].values, axis=1)
    df['Soil_Type'] = np.argmax(df[soil_cols].values, axis=1)
    df = df.drop(columns=wilderness_cols + soil_cols)
    
    numerical_cols = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                      'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                      'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
    categorical_cols = ['Wilderness_Area', 'Soil_Type']
    
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=random_seed, stratify=df['target_bin'])
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    # 2nd least samples in the training set
    minority_class = df_train['Cover_Type'].value_counts().index[-2]
    df_train['S0'] = (df_train['Cover_Type'] == minority_class).astype(int)
    df_test['S0'] = (df_test['Cover_Type'] == minority_class).astype(int)
    
    cat_cardinalities = []
    for col in categorical_cols:
        unique_vals = df_train[col].astype(str).unique()
        val_to_idx = {val: i for i, val in enumerate(unique_vals)}
        oov_idx = len(unique_vals)
        
        df_train[col] = df_train[col].astype(str).map(val_to_idx)
        df_test[col] = df_test[col].astype(str).map(val_to_idx).fillna(oov_idx).astype(int)
        cat_cardinalities.append(oov_idx + 1)
        
    scaler = StandardScaler()
    df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])
    df_test[numerical_cols] = scaler.transform(df_test[numerical_cols])
    
    X_num_train = torch.tensor(df_train[numerical_cols].values.astype(np.float32), dtype=torch.float32)
    X_num_test = torch.tensor(df_test[numerical_cols].values.astype(np.float32), dtype=torch.float32)
    X_cat_train = torch.tensor(df_train[categorical_cols].values.astype(np.int64), dtype=torch.long)
    X_cat_test = torch.tensor(df_test[categorical_cols].values.astype(np.int64), dtype=torch.long)
    y_train = torch.tensor(df_train['target_bin'].values, dtype=torch.long)
    y_test = torch.tensor(df_test['target_bin'].values, dtype=torch.long)
    S0_train = torch.tensor(df_train['S0'].values, dtype=torch.bool)
    S0_test = torch.tensor(df_test['S0'].values, dtype=torch.bool)
    
    train_dataset = TabularDataset(X_num_train, X_cat_train, y_train, S0_train)
    test_dataset = TabularDataset(X_num_test, X_cat_test, y_test, S0_test)
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    evalloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return trainloader, evalloader, testloader, len(numerical_cols), cat_cardinalities


def get_cifar10_dataloaders(batch_size, remove_percentage, random_seed):
    print("Downloading and preparing CIFAR-10 dataset...")
    hf_dataset = load_dataset("uoft-cs/cifar10")
    
    def create_dataset(train=True):
        split = 'train' if train else 'test'
        raw_set = hf_dataset[split]
        targets = np.array(raw_set['label'])
        data = np.stack([np.array(img) for img in raw_set['img']])

        plane_idx, car_idx, other_idx = np.where(targets == 0)[0], np.where(targets == 1)[0], np.where(targets > 1)[0]
        rng = np.random.RandomState(random_seed)
        
        keep_cars_ratio = (100.0 - remove_percentage) / 100.0
        
        if train:
            keep_planes = int(len(plane_idx) * 0.95)
            keep_cars = int(len(car_idx) * keep_cars_ratio)
        else:
            keep_planes = int(len(plane_idx) * 0.50)
            keep_cars = int(len(car_idx) * 0.50)
            
        indices = np.concatenate([rng.choice(plane_idx, keep_planes, False), rng.choice(car_idx, keep_cars, False), other_idx])
        rng.shuffle(indices)

        orig_labels = targets[indices]
        new_labels = np.where(orig_labels <= 1, 0, orig_labels - 1)
        is_original_car = np.where(orig_labels == 1, 1, 0)

        X = torch.tensor(data[indices]).permute(0, 3, 1, 2).float() / 255.0
        Y = torch.tensor(new_labels).long()
        is_car = torch.tensor(is_original_car).bool()

        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1,3,1,1)
        std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1,3,1,1)
        return TensorDataset((X - mean) / std, Y, is_car)

    train_ds = create_dataset(train=True)
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
            DataLoader(train_ds, batch_size=batch_size, shuffle=False),
            DataLoader(create_dataset(train=False), batch_size=batch_size, shuffle=False))

# =========================================================
# 4. UNIFIED TRAINING & METRIC TRACKING 
# =========================================================
def get_split_acc(m, loader, device, is_cifar):
    m.eval()
    correct_s0, total_s0, correct_s1, total_s1 = 0, 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            if is_cifar:
                inputs, labels, s0_mask = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                outputs = m(inputs)
            else:
                x_num, x_cat, labels, s0_mask = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device)
                outputs = m(x_num, x_cat)
                
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, predicted = torch.max(outputs, 1)

            mask_s0 = (s0_mask == True).cpu().numpy()
            mask_s1 = (s0_mask == False).cpu().numpy()
            labels_cpu = labels.cpu().numpy()
            predicted_cpu = predicted.cpu().numpy()

            total_s0 += mask_s0.sum()
            total_s1 += mask_s1.sum()
            
            if mask_s0.sum() > 0:
                correct_s0 += (predicted_cpu[mask_s0] == labels_cpu[mask_s0]).sum()
            if mask_s1.sum() > 0:
                correct_s1 += (predicted_cpu[mask_s1] == labels_cpu[mask_s1]).sum()

    acc_s0 = 100 * correct_s0 / total_s0 if total_s0 > 0 else 0.0
    acc_s1 = 100 * correct_s1 / total_s1 if total_s1 > 0 else 0.0
    return acc_s0, acc_s1


def train_and_track(model, optimizers, trainloader, evalloader, testloader, device, epochs, exp_name, output_dir, is_cifar, txt_path):
    criterion = nn.CrossEntropyLoss()
    if not isinstance(optimizers, list):
        optimizers = [optimizers]
    
    train_acc_s0_history, train_acc_s1_history = [], []
    test_acc_s0_history, test_acc_s1_history = [], []
    
    for epoch in range(epochs + 1):
        if epoch > 0:
            model.train()
            for batch in trainloader:
                if is_cifar:
                    inputs, labels = batch[0].to(device), batch[1].to(device)
                    for opt in optimizers: opt.zero_grad()
                    outputs = model(inputs)
                else:
                    x_num, x_cat, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
                    for opt in optimizers: opt.zero_grad()
                    outputs = model(x_num, x_cat)
                
                if isinstance(outputs, tuple):
                    outputs, M_loss = outputs
                    loss = criterion(outputs, labels) - 1e-3 * M_loss
                else:
                    loss = criterion(outputs, labels)
                    
                loss.backward()
                for opt in optimizers: opt.step()

        train_s0, train_s1 = get_split_acc(model, evalloader, device, is_cifar)
        test_s0, test_s1 = get_split_acc(model, testloader, device, is_cifar)

        train_acc_s0_history.append(train_s0)
        train_acc_s1_history.append(train_s1)
        test_acc_s0_history.append(test_s0)
        test_acc_s1_history.append(test_s1)
        
        print(f"[{exp_name}] Ep {epoch:03d}/{epochs} | Tr S0: {train_s0:5.2f}% | Tr S1: {train_s1:5.2f}% | Te S0: {test_s0:5.2f}% | Te S1: {test_s1:5.2f}%")

    # [Metrics Calculations] 
    # Array mapping: epochs 1-50 are mapped to indices [1:51].
    if len(train_acc_s0_history) > 50:
        early_tr_s0 = np.mean(train_acc_s0_history[1:51])
        early_tr_s1 = np.mean(train_acc_s1_history[1:51])
    else:
        early_tr_s0, early_tr_s1 = 0.0, 0.0
        
    # Array mapping: epochs 280-300 are mapped to indices [280:301].
    if len(test_acc_s0_history) >= 301:
        final_te_s0 = np.mean(test_acc_s0_history[280:301])
        final_te_s1 = np.mean(test_acc_s1_history[280:301])
    else:
        # Fallback block
        final_te_s0 = np.mean(test_acc_s0_history[-21:])
        final_te_s1 = np.mean(test_acc_s1_history[-21:])

    # Log structured text into standard `.txt` document to preserve Arrays and calculations
    with open(txt_path, "a") as f:
        f.write(f"\n=======================================================\n")
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Early Training S0 Accuracy (Epochs 1-50): {early_tr_s0:.2f}%\n")
        f.write(f"Early Training S1 Accuracy (Epochs 1-50): {early_tr_s1:.2f}%\n")
        f.write(f"Final Test S0 Accuracy (Epochs 280-300): {final_te_s0:.2f}%\n")
        f.write(f"Final Test S1 Accuracy (Epochs 280-300): {final_te_s1:.2f}%\n")
        f.write(f"=======================================================\n")
        f.write("Epoch\tTrain_S0\tTrain_S1\tTest_S0\tTest_S1\n")
        for ep in range(epochs + 1):
            f.write(f"{ep}\t{train_acc_s0_history[ep]:.2f}\t{train_acc_s1_history[ep]:.2f}\t{test_acc_s0_history[ep]:.2f}\t{test_acc_s1_history[ep]:.2f}\n")

    # Generate Image Plot visual counterparts mapping identically to text variables
    safe_name = exp_name.replace(" ", "_").lower()
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{exp_name}", fontsize=16)
    
    axs[0,0].plot(train_acc_s0_history, color='tab:blue'); axs[0,0].set_title(f"Train S0: {train_s0:.2f}%")
    axs[0,1].plot(train_acc_s1_history, color='tab:orange'); axs[0,1].set_title(f"Train S1: {train_s1:.2f}%")
    axs[1,0].plot(test_acc_s0_history, color='tab:green'); axs[1,0].set_title(f"Test S0: {test_s0:.2f}%")
    axs[1,1].plot(test_acc_s1_history, color='tab:red'); axs[1,1].set_title(f"Test S1: {test_s1:.2f}%")
    
    for ax in axs.flat:
        ax.grid(True)
        ax.set_ylim(0, 105)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{safe_name}.png"))
    plt.close(fig)


# =========================================================
# 5. MAIN ROUTER & EXECUTION
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    parser = argparse.ArgumentParser(description="Unified Pipeline: Covertype & CIFAR-10")
    parser.add_argument('--dataset', type=str, required=True, choices=['cifar10', 'covertype'], help="Dataset to execute.")
    parser.add_argument('--arch', type=str, default=None, help="CIFAR: vgg19_bn, resnet18... | Covertype: mlp, fttransformer")
    parser.add_argument('--epochs', type=int, default=300, help="Epoch count; safely increased standard to 300.")
    parser.add_argument('--runs', type=int, default=1, help="Execute workflow sequentially this amount of times.")
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--remove-percentage', type=float, default=95.0, help="Ratio of CIFAR-10 Cars to remove.")
    parser.add_argument('--seed', type=int, default=42, help="Base random seed reproducibility identifier.")
    parser.add_argument('--output-dir', type=str, default='./results')
    args = parser.parse_args()

    # Preemptively setup Defaults corresponding natively to Datasets
    if args.arch is None:
        args.arch = 'vgg19_bn' if args.dataset == 'cifar10' else 'mlp'
    if args.batch_size is None:
        args.batch_size = 256 if args.dataset == 'cifar10' else 2048

    os.makedirs(args.output_dir, exist_ok=True)
    txt_log_file = os.path.join(args.output_dir, f"results_log_{args.dataset}_{args.arch}.txt")

    with open(txt_log_file, "w") as f:
        f.write(f"=== EXPERIMENT SUMMARY ===\n")
        f.write(f"Dataset: {args.dataset}\nArchitecture: {args.arch}\nEpochs: {args.epochs}\nRuns: {args.runs}\nBase Seed: {args.seed}\n")
        f.write("==========================\n\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing natively using device: {device}\n")

    for run_idx in range(1, args.runs + 1):
        # Reproducible independent sequence shifts using base index per run
        current_seed = args.seed + run_idx - 1
        set_seed(current_seed)
        print(f"\n{'='*40}\nStarting Target Run {run_idx}/{args.runs} (Seed: {current_seed})\n{'='*40}")

        if args.dataset == 'cifar10':
            train_ldr, eval_ldr, test_ldr = get_cifar10_dataloaders(args.batch_size, args.remove_percentage, current_seed)
            base_model = get_cifar10_model(args.arch, device)
            is_cf = True
        else:
            train_ldr, eval_ldr, test_ldr, num_features, cat_cards = get_covertype_dataloaders(args.batch_size, current_seed)
            if args.arch.lower() == 'fttransformer':
                base_model = FTTransformerWrapper(num_features, cat_cards).to(device)
            else:
                base_model = MLPWrapper(num_features, cat_cards).to(device)
            is_cf = False

        initial_state = copy.deepcopy(base_model.state_dict())

        # Training Sequence Loop: AdamW, SGD, Muon
        model_adamw = copy.deepcopy(base_model)
        model_adamw.load_state_dict(initial_state)
        opt_adamw = optim.AdamW(model_adamw.parameters(), lr=1e-3, weight_decay=1e-2)
        train_and_track(model_adamw, opt_adamw, train_ldr, eval_ldr, test_ldr, device, args.epochs, f"AdamW_run{run_idx}", args.output_dir, is_cf, txt_log_file)

        model_sgd = copy.deepcopy(base_model)
        model_sgd.load_state_dict(initial_state)
        opt_sgd = optim.SGD(model_sgd.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
        train_and_track(model_sgd, opt_sgd, train_ldr, eval_ldr, test_ldr, device, args.epochs, f"SGD_run{run_idx}", args.output_dir, is_cf, txt_log_file)

        model_muon = copy.deepcopy(base_model)
        model_muon.load_state_dict(initial_state)
        muon_params = [p for p in model_muon.parameters() if p.ndim >= 2]
        other_params = [p for p in model_muon.parameters() if p.ndim < 2]
        opt_muon = Muon(muon_params, lr=0.02, momentum=0.95, nesterov=True)
        opt_other = optim.AdamW(other_params, lr=1e-3, weight_decay=1e-2)
        train_and_track(model_muon, [opt_muon, opt_other], train_ldr, eval_ldr, test_ldr, device, args.epochs, f"Muon_run{run_idx}", args.output_dir, is_cf, txt_log_file)

if __name__ == '__main__':
    main()
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
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, f"{exp_name.replace(' ', '_').lower()}.png"))
    plt.close(fig)

    # =========================================================
    # NEW: TEXT LOGGING BLOCK
    # =========================================================
    log_file_path = os.path.join(output_dir, f"{exp_name.replace(' ', '_').lower()}_logs.txt")
    with open(log_file_path, "w") as f:
        if is_cifar:
            f.write("Epoch\tTrain_S0_Acc\tTrain_S1_Acc\tTest_S0_Acc\tTest_S1_Acc\n")
            for i in range(len(metrics['train_acc_s0'])):
                f.write(f"{i}\t{metrics['train_acc_s0'][i]:.2f}\t{metrics['train_acc_s1'][i]:.2f}\t{metrics['test_acc_s0'][i]:.2f}\t{metrics['test_acc_s1'][i]:.2f}\n")
        else:
            f.write("Epoch\tTrain_Acc\tTest_Acc\n")
            for i in range(len(metrics['train_acc_s0'])):
                f.write(f"{i}\t{metrics['train_acc_s0'][i]:.2f}\t{metrics['test_acc_s0'][i]:.2f}\n")
    print(f"[{exp_name}] Accuracy logs saved to: {log_file_path}")
    # =========================================================

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
