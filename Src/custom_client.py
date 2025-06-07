from typing import Dict, Tuple, List, Optional
import torch
import torch.nn as nn
import numpy as np
from flwr.common import (
    GetPropertiesIns, GetPropertiesRes,
    GetParametersIns, GetParametersRes,
    FitIns, FitRes, EvaluateIns, EvaluateRes,
    Parameters, ndarrays_to_parameters,
    parameters_to_ndarrays, Scalar, Code, Status
)
from flwr.client import Client
from torch.utils.data import DataLoader
import json

class CustomClient(Client):
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        strategy_type: str = "fedavg",
        mu: float = 0.1,
        learning_rate: float = 0.01,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.strategy_type = strategy_type
        self.mu = mu
        self.learning_rate = learning_rate
        self.local_control: Optional[List[np.ndarray]] = None
        self.local_model_state: Optional[List[np.ndarray]] = None

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        return GetPropertiesRes(
            status=Status(code=Code.OK, message="Success"),
            properties={}
        )

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        parameters = ndarrays_to_parameters(self.model.get_model_parameters())
        return GetParametersRes(
            status=Status(code=Code.OK, message="Success"),
            parameters=parameters
        )

    def fit(self, ins: FitIns) -> FitRes:
        try:
            # Load model parameters
            global_params = parameters_to_ndarrays(ins.parameters)
            self.model.set_model_parameters(global_params)

            metrics = {}
            updated_params = None  # Initialize to ensure it's always defined

            if self.strategy_type == "fedprox":
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
                loss, accuracy = self.model.train_epoch_prox(
                    self.train_loader, criterion, optimizer,
                    self.device, mu=self.mu, global_params=global_params
                )
                metrics = {
                    "train_loss": loss,
                    "train_accuracy": accuracy
                }

            elif self.strategy_type == "scaffold":
                # Parse control variates
                c_global = [np.array(arr) for arr in json.loads(ins.config["global_control"])]
                c_local = [np.array(arr) for arr in json.loads(ins.config["client_control"])]

                if self.local_control is None:
                    self.local_control = c_local

                self.local_model_state = self.model.get_model_parameters()

                # Train with SCAFFOLD
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
                
                loss, accuracy, updated_control = self.model.train_epoch_scaffold(
                    train_loader=self.train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=self.device,
                    global_control=c_global,
                    local_control=self.local_control,
                    lr=self.learning_rate
                )

                self.local_control = updated_control
                metrics = {
                    "train_loss": loss,
                    "train_accuracy": accuracy,
                    "client_control_update": json.dumps([arr.tolist() for arr in updated_control])
                }

            else:  # fedavg
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
                loss, accuracy = self.model.train_epoch(
                    self.train_loader, criterion, optimizer, self.device
                )
                metrics = {
                    "train_loss": loss,
                    "train_accuracy": accuracy
                }

            # Get updated parameters (must be after all training branches)
            updated_params = ndarrays_to_parameters(self.model.get_model_parameters())

            return FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=updated_params,
                num_examples=len(self.train_loader.dataset),
                metrics=metrics
            )

        except Exception as e:
            return FitRes(
                status=Status(code=Code.FIT_ERROR, message=str(e)),
                parameters=None,
                num_examples=0,
                metrics={}
            )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        try:
            parameters = parameters_to_ndarrays(ins.parameters)
            self.model.set_model_parameters(parameters)

            criterion = nn.CrossEntropyLoss()
            loss, accuracy = self.model.test_epoch(self.test_loader, criterion, self.device)

            return EvaluateRes(
                status=Status(code=Code.OK, message="Success"),
                loss=float(loss),
                num_examples=len(self.test_loader.dataset),
                metrics={"val_accuracy": accuracy}
            )
        except Exception as e:
            return EvaluateRes(
                status=Status(code=Code.EVALUATE_ERROR, message=str(e)),
                loss=0.0,
                num_examples=0,
                metrics={}
            )

    def to_client(self) -> 'CustomClient':
        return self