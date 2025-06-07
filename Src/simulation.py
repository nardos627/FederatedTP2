import flwr as fl
import json
import torch
from .run_client import client_fn
from .results_visualizer import ResultsVisualizer


def convert_metrics(metrics_dict):
    if not metrics_dict or not isinstance(metrics_dict, dict):
        return []
    rounds = sorted(set(r for metric_list in metrics_dict.values() for r, _ in metric_list))
    result_list = []
    for r in rounds:
        metric_dict = {}
        for name, metric_list in metrics_dict.items():
            val = next((v for round_i, v in metric_list if round_i == r), None)
            if val is not None:
                metric_dict[name] = val
        result_list.append((r, metric_dict))
    return result_list


def run_simulation(num_clients: int = 2, num_rounds: int = 3) -> None:
   

 
    hist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        client_resources={"num_cpus": 1, "num_gpus": 0.5 if torch.cuda.is_available() else 0}
    )

    
    results = {
        "losses_distributed": hist.losses_distributed if hasattr(hist, "losses_distributed") else [],
        "metrics_distributed": convert_metrics(getattr(hist, "metrics_distributed", {})),
        "metrics_distributed_fit": convert_metrics(getattr(hist, "metrics_distributed_fit", {}))
    }

   
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("✅ Simulation completed. Results saved to results.json")

    
    visualizer = ResultsVisualizer()
    visualizer.load_simulation_results("results.json")
    visualizer.plot_results("./figures")
    visualizer.print_results_table()
    print("📊 Visualization complete. Plots saved in ./figures/")
