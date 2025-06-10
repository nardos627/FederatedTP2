# Federated Learning: Handling Data Heterogeneity and Client Drift

![Github Repo](https://github.com/nardos627/FederatedTP2) 


## 📝 Description
Experimental analysis of federated learning algorithms (FedAvg, FedProx, SCAFFOLD) under varying data heterogeneity conditions (α=10, 1.0, 0.1). Investigates client drift mitigation and comparative algorithm performance.

## 🚀 Quick Start Commands

### Server Execution
```bash
# Standard FedAvg
python main.py run-server --strategy fedavg --rounds 50

# FedProx with μ=0.1
python main.py run-server --strategy fedprox --mu 0.1 --rounds 50

# SCAFFOLD implementation
python main.py run-server --strategy scaffold --rounds 50

# FedAvg client
python main.py run-client --cid=0 --strategy=fedavg

# FedProx client 
python main.py run-client --cid=0 --strategy=fedprox

# SCAFFOLD client
python main.py run-client --cid=0 --strategy=scaffold

## 📊 Experimental Results

### Accuracy by Heterogeneity Level
| Algorithm   | α=10 (Near-IID) | α=1.0 (Moderate) | α=0.1 (High) |
|-------------|-----------------|------------------|--------------|
| FedAvg      | 85%             | 70%              | 55%          |
| FedProx     | 84%             | 79%              | 71%          |
| SCAFFOLD    | 86%             | 81%              | 78%          |

### Performance Characteristics
- **Convergence Speed**: SCAFFOLD (Fastest) > FedProx > FedAvg (Slowest)
- **Client Drift Resistance**: SCAFFOLD (Best) > FedProx > FedAvg
- **Computation Cost**: SCAFFOLD (Highest) > FedProx > FedAvg (Lowest)

## 🔍 Key Findings
- SCAFFOLD dominates across all heterogeneity levels (+5-23% over FedAvg)
- FedProx most effective at μ=0.1 (optimal regularization)
- FedAvg only suitable for near-IID data (α≥1.0)
