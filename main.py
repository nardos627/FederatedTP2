import argparse
from Src.data_utils import generate_distributed_datasets
from Src.server import run_server
from Src.results_visualizer import ResultsVisualizer
from Src.run_client import run_client

def main():
    parser = argparse.ArgumentParser(description="Federated Learning TP1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Data generation command
    data_parser = subparsers.add_parser("generate-data")
    data_parser.add_argument("--num-clients", type=int, default=10)
    data_parser.add_argument("--alpha", type=float, default=1.0)
    data_parser.add_argument("--data-dir", type=str, default="./data/client_data")
    
    # Server command
    server_parser = subparsers.add_parser("run-server")
    server_parser.add_argument("--address", type=str, default="127.0.0.1:8080")
    server_parser.add_argument("--rounds", type=int, default=3)
    server_parser.add_argument("--output", type=str, default="results.json")
    server_parser.add_argument("--strategy", type=str, default="fedavg",
                             choices=["fedavg", "fedprox", "scaffold"],
                             help="Federated learning strategy")
    server_parser.add_argument("--mu", type=float, default=0.1, help="FedProx proximal term weight")
    # Client command
    client_parser = subparsers.add_parser("run-client")
    client_parser.add_argument("--cid", type=int, required=True, help="Client ID (0 to num-clients-1)")
    client_parser.add_argument("--strategy", type=str, default="fedavg",
                            choices=["fedavg", "fedprox", "scaffold"],
                            help="Federated learning strategy")
    client_parser.add_argument("--mu", type=float, default=0.1,
                            help="FedProx proximal term weight")
    client_parser.add_argument("--lr", type=float, default=0.01,
                            help="Learning rate used for training")

    
    # Simulation command
    sim_parser = subparsers.add_parser("run-simulation")
    sim_parser.add_argument("--num-clients", type=int, default=2)
    sim_parser.add_argument("--rounds", type=int, default=3)
    
    # Visualization command
    vis_parser = subparsers.add_parser("visualize")
    vis_parser.add_argument("--results-file", type=str, default="results.json")
    vis_parser.add_argument("--output-dir", type=str, default="./figures")

    
    
    
    args = parser.parse_args()
    
    if args.command == "generate-data":
        generate_distributed_datasets(args.num_clients, args.alpha, args.data_dir)
        print(f"Generated datasets for {args.num_clients} clients")

    elif args.command == "run-server":
        run_server(
            server_address=args.address,
            num_rounds=args.rounds,
            output_file=args.output,
            strategy_type=args.strategy,
            mu=args.mu
        )

    elif args.command == "run-client":
        run_client(args.cid, args.strategy, args.mu)

    elif args.command == "run-simulation":
        from Src.simulation import run_simulation
        run_simulation(args.num_clients, args.rounds)

    elif args.command == "visualize":
        visualizer = ResultsVisualizer()
        visualizer.load_simulation_results(args.results_file)
        visualizer.plot_results(args.output_dir)
     

    elif args.command == "simulate-and-visualize":
        from Src.simulation import run_simulation

        # Run simulation
        run_simulation(args.num_clients, args.rounds)

        # Visualize
        visualizer = ResultsVisualizer()
        visualizer.load_simulation_results(args.results_file)
        visualizer.plot_results(args.output_dir)
        visualizer.print_results_table()

if __name__ == "__main__":
    main()
