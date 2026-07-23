import os
import csv  # <-- Added: import the built-in csv module
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def get_split_acc(m, loader, device):
    m.eval()
    correct_s0, total_s0, correct_s1, total_s1 = 0, 0, 0, 0
    with torch.no_grad():
        for x_num, x_cat, labels, s0_mask in loader:
            x_num, x_cat, labels = x_num.to(device), x_cat.to(device), labels.to(device)
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

def train_and_track(model, optimizers, trainloader, testloader, device, epochs=200, experiment_name="Training", output_dir="results"):
    criterion = nn.CrossEntropyLoss()
    if not isinstance(optimizers, list):
        optimizers = [optimizers]

    test_acc_s0_history, test_acc_s1_history = [], []

    for epoch in range(epochs + 1):
        if epoch > 0:
            model.train()
            for x_num, x_cat, labels, _ in trainloader:
                x_num, x_cat, labels = x_num.to(device), x_cat.to(device), labels.to(device)
                for opt in optimizers: 
                    opt.zero_grad()
                    
                outputs = model(x_num, x_cat)
                if isinstance(outputs, tuple):
                    outputs, M_loss = outputs
                    loss = criterion(outputs, labels) - 1e-3 * M_loss
                else:
                    loss = criterion(outputs, labels)
                    
                loss.backward()
                for opt in optimizers: 
                    opt.step()

        test_s0, test_s1 = get_split_acc(model, testloader, device)
        test_acc_s0_history.append(test_s0)
        test_acc_s1_history.append(test_s1)
        
        print(f"[{experiment_name}] Epoch {epoch:03d}/{epochs} | Test Acc S0: {test_s0:5.2f}% | Test Acc S1: {test_s1:5.2f}%")

    # ---------------------------------------------------------
    # Setup directory and filenames
    # ---------------------------------------------------------
    safe_name = experiment_name.replace(" ", "_").lower()
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # Save the CSV logs (NEW CODE)
    # ---------------------------------------------------------
    csv_path = os.path.join(output_dir, f"{safe_name}_logs.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Test_Acc_S0", "Test_Acc_S1"])  # Header row
        for epoch_idx in range(len(test_acc_s0_history)):
            writer.writerow([epoch_idx, test_acc_s0_history[epoch_idx], test_acc_s1_history[epoch_idx]])
    
    print(f"Saved training logs to {csv_path}")

    # ---------------------------------------------------------
    # Save final plot after training
    # ---------------------------------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{experiment_name} | Final Epoch {epochs}", fontsize=16, fontweight='bold')

    axs[0].plot(test_acc_s0_history, color='tab:green', linewidth=2)
    axs[0].set_title(f"Test S0 (Minority): {test_s0:.2f}%", fontsize=12)

    axs[1].plot(test_acc_s1_history, color='tab:red', linewidth=2)
    axs[1].set_title(f"Test S1 (Majority): {test_s1:.2f}%", fontsize=12)

    for ax in axs:
        ax.set_ylabel("Accuracy (%)")
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 105)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f"{safe_name}_plot.png")
    plt.savefig(plot_path)
    print(f"Saved final plot to {plot_path}")
    plt.close(fig)

    return test_acc_s0_history, test_acc_s1_history
