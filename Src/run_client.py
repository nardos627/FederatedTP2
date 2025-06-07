import argparse
from .data_utils import load_client_data
from .customfashionmodel import CustomFashionModel
from .custom_client import CustomClient
import torch
import flwr as fl

def run_client(cid: int, strategy_type: str = "fedavg", mu: float = 0.1) -> None:
    """Run a single client instance."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader = load_client_data(
        cid=cid,
        data_dir="./data/client_data",
        batch_size=32
    )
    
    model = CustomFashionModel().to(device)
    client = CustomClient(
        model, 
        train_loader, 
        val_loader, 
        device,
        strategy_type=strategy_type,
        mu=mu,
       
        
    )
 
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client.to_client()
    )

def client_fn(cid: str, strategy_type: str = "fedavg", mu: float = 0.1):
    """Return a Flower client instance for a given client ID (used in simulation)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = load_client_data(
        cid=int(cid),
        data_dir="./data/client_data",
        batch_size=32
    )

    model = CustomFashionModel().to(device)
    client = CustomClient(
        model, 
        train_loader, 
        val_loader, 
        device,
        strategy_type=strategy_type,
        mu=mu
    )
    return client.to_client()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FL client")
    parser.add_argument("--cid", type=int, required=True, help="Client ID")
    parser.add_argument("--strategy", type=str, default="fedavg", 
                       choices=["fedavg", "fedprox", "scaffold"], 
                       help="Federated learning strategy")
    parser.add_argument("--mu", type=float, default=0.1, 
                       help="Proximal term weight for FedProx")
    args = parser.parse_args()
    
    print(f"Starting client {args.cid} with strategy {args.strategy}...")
    run_client(args.cid, args.strategy, args.mu)