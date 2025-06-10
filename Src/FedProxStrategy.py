from .Fedstrategy import FedAvgStrategy
from flwr.common import Parameters, FitIns, FitRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Dict, Optional, Union
import numpy as np
import time
from .client_manager import ClientManager
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from flwr.server.client_proxy import ClientProxy

class FedProxStrategy(FedAvgStrategy):
    def __init__(self, initial_parameters: Optional[Parameters] = None, mu: float = 0.1):
        super().__init__(initial_parameters)
        self.mu = mu  # Proximal term weight

    def initialize_parameters(self, client_manager: ClientManager) -> Optional[Parameters]:
        print("Waiting for at least 1 client to connect...")
        timeout = 60 
        start_time = time.time()
        
        while client_manager.num_available() < 1:
            if time.time() - start_time > timeout:
                raise RuntimeError("Timeout: No clients connected to server.")
            time.sleep(1)
            print(f"{client_manager.num_available()} clients connected. Waiting...")
        
        print(f"{client_manager.num_available()} clients connected. Proceeding with initialization.")
        return self.initial_parameters

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # Sample clients using FedAvg logic
        sampled_clients = super().configure_fit(server_round, parameters, client_manager)

       
        config = {
            "server_round": server_round,
            "epochs": 3,
            "batch_size": 64,
            "learning_rate": 0.01,
            "mu": self.mu
        }

       
        return [(client, FitIns(parameters=parameters, config=config)) for client, _ in sampled_clients]
