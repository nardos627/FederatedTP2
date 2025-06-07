import os
import json
import matplotlib.pyplot as plt
from typing import Dict


def load_results(file_path: str) -> Dict:
    with open(file_path, "r") as f:
        content = f.read().strip()
        if not content:
            raise ValueError(f"{file_path} is empty.")
        return json.loads(content)


def compare_multiple_results(file_label_map: Dict[str, str], fig_dir: str = "./figures_comparecl") -> None:
    os.makedirs(fig_dir, exist_ok=True)
    all_results = {}

    for file_path, label in file_label_map.items():
        all_results[label] = load_results(file_path)

    plt.figure(figsize=(8, 5))
    for label, data in all_results.items():
        rounds = [r[0] for r in data.get("losses_distributed", [])]
        losses = [r[1] for r in data.get("losses_distributed", [])]
        plt.plot(rounds, losses,label=f"Loss - {label}")
    plt.title("Loss per Round (Comparison)")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "compare_loss.png"))
    plt.close()

   
    plt.figure(figsize=(8, 5))
    for label, data in all_results.items():
        rounds = [r[0] for r in data.get("metrics_distributed", [])]
        val_acc = [r[1].get("val_accuracy", 0.0) for r in data.get("metrics_distributed", [])]
        plt.plot(rounds, val_acc, label=f"Val Acc - {label}")
    plt.title("Validation Accuracy per Round (Comparison)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "compare_val_accuracy.png"))
    plt.close()

    
    plt.figure(figsize=(8, 5))
    for label, data in all_results.items():
        rounds = [r[0] for r in data.get("metrics_distributed_fit", [])]
        train_acc = [r[1].get("train_accuracy", 0.0) for r in data.get("metrics_distributed_fit", [])]
        plt.plot(rounds, train_acc, label=f"Train Acc - {label}")
    plt.title("Training Accuracy per Round (Comparison)")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(fig_dir, "compare_train_accuracy.png"))
    plt.close()


if __name__ == "__main__":
 
    file_label_map = {
        "results30c5.json": "Num-Client=5",
        "results30.json": "Num-Clients=10",
        "results30c15.json": "Num-Clients=15"
    }

    compare_multiple_results(file_label_map)
