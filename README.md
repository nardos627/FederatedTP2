# Federated Learning with Flower Framework - TP1

## Project Overview
Implementation of a Federated Learning system using Flower framework with Fashion MNIST dataset. The project compares different FL configurations through:

- A **CNN model** (2 convolutional layers + FC layers with dropout)
- Multiple client-server simulations
- Hyperparameter impact analysis

## Key Experiments

### 1. Training Rounds Comparison
| Rounds | Final Accuracy |
|--------|----------------|
| 3      | 0.82%       | 
| 10     | 0.84%       | 
| 30     | 0.85%       | 



### 2. Batch Size Impact
| Batch Size | Accuracy |
|------------|----------|
| 16         | 0.82% | 
| 32         | 0.85% |
| 64         | 0.88% | 



### 3. Learning Rate Variations
| LR    | Accuracy | 
|-------|----------|
| 0.001 | 0.87% | 
| 0.01  | 0.85% | 
| 0.1   | 0.76% | 



### 4. Data Heterogeneity (α)
| α   | Accuracy | 
|-----|----------|
| 0.1 | 0.30% | 
| 1   | 0.85% | 
| 10  | 0.84% | 


## How to Run
```bash
# Generate datasets
python main.py generate_data --num-clients 10  --alpha 1.0

# Start server
python main.py run_server --rounds 30

# Start clients (in separate terminals)
python run_client.py --cid 1
python run_client.py --cid 2
