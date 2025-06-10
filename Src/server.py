import flwr as fl
from flwr.server import Server
from flwr.server.history import History
from typing import Optional, Dict, List, Tuple
import json
from flwr.server.strategy import Strategy
from flwr.server.client_manager import ClientManager
from .Fedstrategy import FedAvgStrategy
from .FedProxStrategy import FedProxStrategy
from .client_manager import CustomClientManager
from .Fedscaffold import SCAFFOLDStrategy

def convert_metrics(metrics: Dict[str, List[tuple[int, float]]]) -> List[tuple[int, Dict[str, float]]]:
    converted: Dict[int, Dict[str, float]] = {}
    for metric_name, metric_list in metrics.items():
        for round_num, val in metric_list:
            if round_num not in converted:
                converted[round_num] = {}
            converted[round_num][metric_name] = float(val)
    return sorted(converted.items())

def save_results(history: History, filename: str = "results.json") -> None:
    results = {
        "losses_distributed": [(rnd, float(loss)) for rnd, loss in history.losses_distributed],
        "metrics_distributed": convert_metrics(getattr(history, "metrics_distributed", {})),
        "metrics_distributed_fit": convert_metrics(getattr(history, "metrics_distributed_fit", {})),
       
    }

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    print(f"✅ Results saved to {filename}")

def run_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 3,
    strategy: Optional[Strategy] = None,
    client_manager: Optional[ClientManager] = None,
    strategy_type: str = "scaffold",
    output_file: str = "results.json",
    mu: float = 0.1  # Add mu parameter
) -> History:
   
    config = fl.server.ServerConfig(num_rounds=num_rounds)

    if strategy is None:
        if strategy_type == "fedavg":
            strategy = FedAvgStrategy()
        elif strategy_type == "fedprox":
            strategy = FedProxStrategy(mu=mu)
        elif strategy_type == "scaffold":
            strategy = SCAFFOLDStrategy()
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
              
    if client_manager is None:
        client_manager = CustomClientManager()

    print(f"Starting server on {server_address} for {num_rounds} rounds...")
    history = fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
        client_manager=client_manager
    )

    
    save_results(history, output_file)

    return history
