"""
Hierarchical FL (clients -> fog node → cloud)

Metrics:
- Accuracy
- Communication rounds
- Bandwidth saved
- Training time
"""

import time
import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from hierarchical.task import Net, load_centralized_dataset, test
from hierarchical.custom_strategy import HierarchicalFedAvg

# Create ServerApp
app = ServerApp()
round_start: float
last_accuracy: float

@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    global round_start
    global last_accuracy

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    num_fog_nodes: int = context.run_config["num-fog-nodes"]
    lr: float = context.run_config["learning-rate"]

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize HierarchicalFedAvg strategy
    strategy = HierarchicalFedAvg(num_fog_nodes=num_fog_nodes, fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg for `num_rounds`
    total_start = round_start = time.perf_counter()
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )
    total_time = time.perf_counter() - total_start

    # Calculate bandwidth saved
    model_bytes = arrays.count_bytes()
    num_clients = len(list(grid.get_node_ids()))
    num_evaluate = int(fraction_evaluate * num_clients)
    hfl_bandwidth = num_rounds * model_bytes * (2 * num_fog_nodes + 2 * num_clients + num_evaluate)
    baseline_bandwidth = 50000 * (32 * 32 * 3 + 1)
    bandwidth_saved = (1 - hfl_bandwidth / baseline_bandwidth) * 100

    print(f"Accuracy: {last_accuracy * 100:.2f}%")
    print(f"Communication rounds: {num_rounds}")
    print(f"Baseline bandwidth: {baseline_bandwidth / 1e6:.2f} MB")
    print(f"HFL bandwidth: {hfl_bandwidth / 1e6:.2f} MB")
    print(f"Bandwidth saved: {bandwidth_saved:.2f}%")
    print(f"Training time: {total_time:.2f} s")

    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")

def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""
    global round_start
    global last_accuracy

    # Calculate the current round time
    round_time = time.perf_counter() - round_start

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Record start time of next round
    round_start = time.perf_counter()

    # Record last accuracy
    last_accuracy = test_acc

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss, "training_time": round_time})
