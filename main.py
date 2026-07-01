import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np

from data import get_dataloaders
from model import FTTransformerWrapper, MLPWrapper
from optimizer import Muon
from train import train_and_track

def evaluate_results(experiment_results, num_epochs):
    start_idx_test = max(0, num_epochs - 19)
    end_idx_test = num_epochs + 1
    opts = ['FT_Transformer_AdamW', 'FT_Transformer_SGD', 'FT_Transformer_Muon', 'MLP_AdamW', 'MLP_SGD', 'MLP_Muon']

    output = []
    def process_analysis(test_key, label, start_idx, end_idx, title):
        output.append(f"--- {title} ---")
        table_data = []
        for opt in opts:
            if opt in experiment_results:
                history = experiment_results[opt][test_key]
                if len(history) >= end_idx - 1:
                    avg_acc = np.mean(history[start_idx:end_idx])
                    table_data.append({'Model & Optimizer': opt, f'Avg {label} Accuracy (%)': round(avg_acc, 2)})
                else:
                    table_data.append({'Model & Optimizer': opt, f'Avg {label} Accuracy (%)': 'Not enough epochs'})
        
        df = pd.DataFrame(table_data)
        output.append(df.to_string(index=False))
        output.append("\n")

    process_analysis('test_s0', 'S0', start_idx_test, end_idx_test, f"Average Test S0 Accuracy (Epochs {start_idx_test}-{end_idx_test-1})")
    process_analysis('test_s0', 'S0', 1, 51, "Average Test S0 Accuracy (Early Epochs 1-50)")
    process_analysis('test_s1', 'S1', start_idx_test, end_idx_test, f"Average Test S1 Accuracy (Epochs {start_idx_test}-{end_idx_test-1})")
    process_analysis('test_s1', 'S1', 1, 51, "Average Test S1 Accuracy (Early Epochs 1-50)")
    
    return "\n".join(output)

def main():
    torch.manual_seed(51)
    np.random.seed(51)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(51)
        torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # CONFIGURATION
    BATCH_SIZE = 2048
    NUM_EPOCHS = 200
    OUTPUT_DIR = "results"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    trainloader, testloader, num_features, cat_cardinalities = get_dataloaders(batch_size=BATCH_SIZE)
    experiment_results = {}

    experiments = [
        ("FT_Transformer_AdamW", FTTransformerWrapper, lambda p: optim.AdamW(p, lr=1e-3, weight_decay=1e-4)),
        ("FT_Transformer_SGD", FTTransformerWrapper, lambda p: optim.SGD(p, lr=1e-2, momentum=0.9, weight_decay=1e-4)),
        ("FT_Transformer_Muon", FTTransformerWrapper, lambda p: Muon(p, lr=0.02)),
        ("MLP_AdamW", MLPWrapper, lambda p: optim.AdamW(p, lr=1e-3, weight_decay=1e-4)),
        ("MLP_SGD", MLPWrapper, lambda p: optim.SGD(p, lr=1e-2, momentum=0.9, weight_decay=1e-4)),
        ("MLP_Muon", MLPWrapper, lambda p: Muon(p, lr=0.02))
    ]

    # Run through the pipeline configuration
    for name, ModelClass, opt_init in experiments:
        print(f"\n=== Training {name} ===")
        model = ModelClass(num_features, cat_cardinalities).to(device)
        optimizer = opt_init(model.parameters())
        
        te_s0, te_s1 = train_and_track(
            model, optimizer, trainloader, testloader, device, 
            epochs=NUM_EPOCHS, experiment_name=name, output_dir=OUTPUT_DIR
        )
        experiment_results[name] = {'test_s0': te_s0, 'test_s1': te_s1}

    # Save dictionary arrays
    torch.save(experiment_results, os.path.join(OUTPUT_DIR, 'all_experiments_accuracies.pt'))
    print(f"\nSaved all training accuracies to '{OUTPUT_DIR}/all_experiments_accuracies.pt'\n")

    # Final Output Metrics Analysis
    print("Evaluating results and generating output tables...")
    results_str = evaluate_results(experiment_results, NUM_EPOCHS)
    print(results_str)
    
    with open(os.path.join(OUTPUT_DIR, 'metrics_tables.txt'), 'w') as f:
        f.write(results_str)

    print(f"Finished! Figures saved as PNGs and tables saved to '{OUTPUT_DIR}/metrics_tables.txt'.")

if __name__ == '__main__':
    main()