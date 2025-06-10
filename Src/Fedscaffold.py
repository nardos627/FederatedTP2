from flwr.server.strategy import Strategy
from flwr.common import (
    Parameters, FitIns, FitRes, EvaluateIns, EvaluateRes,
    ndarrays_to_parameters, parameters_to_ndarrays,
    Scalar, GetParametersIns
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import time
import json



class SCAFFOLDStrategy(Strategy):
    def __init__(
        self,
        initial_parameters: Optional[Parameters] = None,
        server_learning_rate: float = 0.01,
        fraction_fit: float = 0.4, 
        fraction_evaluate: float = 0.3,  
        min_fit_clients: int = 1, 
        min_evaluate_clients: int = 2, 
        min_available_clients: int = 1,  
    ):
        super().__init__()
        self.initial_parameters = initial_parameters
        self.server_learning_rate = server_learning_rate
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        
        # SCAFFOLD specific variables
        self.c_global = None
        self.initialized = False
        self.client_controls: Dict[str, List[np.ndarray]] = {}

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        print("Waiting for at least 1 client to connect...")
        timeout = 60
        start_time = time.time()
        
        while client_manager.num_available() < self.min_available_clients:
            if time.time() - start_time > timeout:
                raise RuntimeError("Timeout: No clients connected to server.")
            time.sleep(1)
            print(f"{client_manager.num_available()} clients connected. Waiting for more...")
        
        print(f"{client_manager.num_available()} clients connected. Proceeding with initialization.")

        if not self.initialized:
            if self.initial_parameters is None:
                client = next(iter(client_manager.all().values()))
                print("Fetching initial parameters from client...")
                parameters_res = client.get_parameters(
                    ins=GetParametersIns(config={}),
                    timeout=10,
                    group_id="0"
                )
                self.initial_parameters = parameters_res.parameters
                print("Initial parameters fetched.")

            # Initialize global control variate with zeros
            params = parameters_to_ndarrays(self.initial_parameters)
            self.c_global = [np.zeros_like(p) for p in params]
            print("Global control variate initialized.")
            self.initialized = True

        return self.initial_parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        if not self.initialized or self.c_global is None:
            self.initialize_parameters(client_manager)

        # Consistent sampling with FedAvg
        sample_size = max(self.min_fit_clients, int(self.fraction_fit * client_manager.num_available()))
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_fit_clients,
            criterion=None
        )

        fit_ins_list = []
        for client in clients:
            cid = client.cid
            
            
            if cid not in self.client_controls:
                self.client_controls[cid] = [np.zeros_like(arr) for arr in self.c_global]

            # Prepare config with control variates
            config = {
                "server_round": server_round,
                "epochs": 3, 
                "batch_size": 64,
                "learning_rate": 0.01,
                "global_control": json.dumps([arr.tolist() for arr in self.c_global]),
                "client_control": json.dumps([arr.tolist() for arr in self.client_controls[cid]]),
            }

            fit_ins = FitIns(parameters=parameters, config=config)
            fit_ins_list.append((client, fit_ins))

        return fit_ins_list

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}

        # Extract weights and example counts
        weights_results = []
        for _, fit_res in results:
            client_weights = parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            weights_results.append((client_weights, num_examples))

        # Calculate weighted average of parameters
        weights_prime = []
        for i in range(len(weights_results[0][0])):
            layer_weights = []
            layer_examples = []
            for client_weights, num_examples in weights_results:
                layer_weights.append(client_weights[i])
                layer_examples.append(num_examples)
            
            weighted_sum = np.sum(
                [w * n for w, n in zip(layer_weights, layer_examples)],
                axis=0
            )
            avg_layer = weighted_sum / sum(layer_examples)
            weights_prime.append(avg_layer)

        # Process control variate updates
        client_control_updates = []
        for client, fit_res in results:
            if "client_control_update" in fit_res.metrics:
                try:
                    update = [np.array(arr) for arr in json.loads(fit_res.metrics["client_control_update"])]
                    client_control_updates.append(update)
                    self.client_controls[client.cid] = update
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Error processing control update from client {client.cid}: {e}")

        # Update global control variate
        if client_control_updates:
            example_counts = [fit_res.num_examples for _, fit_res in results]
            total_examples = sum(example_counts)
            
            avg_client_control_update = [
                sum(
                    update[i] * n for update, n in zip(client_control_updates, example_counts)
                ) / total_examples
                for i in range(len(client_control_updates[0]))
            ]
            
            self.c_global = [
                c_global + (avg_update * self.server_learning_rate)
                for c_global, avg_update in zip(self.c_global, avg_client_control_update)
            ]

        # Aggregate metrics
        metrics_aggregated = {
            "train_loss": np.mean([res.metrics.get("train_loss", 0.0) for _, res in results]),
            "train_accuracy": np.mean([res.metrics.get("train_accuracy", 0.0) for _, res in results]),
        }

        return ndarrays_to_parameters(weights_prime), metrics_aggregated

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Consistent evaluation sampling with FedAvg"""
        available_clients = client_manager.num_available()
        if available_clients == 0:
            return []

        sample_size = max(
            self.min_evaluate_clients,
            int(self.fraction_evaluate * available_clients)
        )
        
        eval_clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_evaluate_clients,
            criterion=None
        )
        
        config = {
            "server_round": server_round,
            "batch_size": 64
        }
        
        evaluate_ins = EvaluateIns(parameters=parameters, config=config)
        return [(client, evaluate_ins) for client in eval_clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        total_examples = sum(res.num_examples for _, res in results)
        weighted_loss = sum(res.loss * res.num_examples for _, res in results) / total_examples
        avg_accuracy = np.mean([res.metrics.get("val_accuracy", 0.0) for _, res in results])

        return float(weighted_loss), {
            "val_accuracy": float(avg_accuracy)
        }

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None