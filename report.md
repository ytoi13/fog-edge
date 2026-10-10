# Standard FL (clients -> cloud server)

All changes are based on the official `@flwrlabs/quickstart-pytorch` example.

https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html

What it does:

> ... train a Convolutional Neural Network on CIFAR-10 using Flower and PyTorch.

Its behavior aligns with standard FL:

> Standard FL (clients -> cloud server)

`server_app.py` calls `FedAvg` only once. `FedAvg` can be roughly understood as dispatching tasks to its immediate downstream clients via a specific algorithm.

`client_app.py` is only responsible for training, and it happens to be the direct subordinate mentioned above.

Our task is to record:

> - Accuracy
> - Communication rounds
> - Bandwidth saved
> - Training time

Accuracy has already been implemented in the original version and output order corresponds to the communication rounds.

## Training time

The process of a training round is:

> Server calls `FedAvg` -> Client receives and completes the task -> Sent back to server

### Single client time

In `client_app.py`, add timers before and after the training function to track single-client training duration.

### Total training time

Add timers before and after `strategy.start`.

### Single round time

Given that the execution logic of the program:

> Train R1 -> Eval R1 -> Train R2 -> Eval R2 -> ...

Reloading the model during each evaluation is time-consuming, and the single-round training time should not include this loading time.

Inside the evaluation function, we can capture the end time before loading the model, and reset the start time right after. The start time can be managed via a global variable.

Then it becomes:

> Log Epoch 1 Start ➔ Train Epoch 1 ➔ Log Epoch 1 End ➔ Reload Model ➔ Log Epoch 2 Start ➔ Train Epoch 2 ➔ ...

## Bandwidth saved

Federated learning bandwidth $B_{FL}$ is:
$$
B_{\rm FL}=R\times S\times(2N+N_e)
$$
where:

* $R$: Training rounds
* $S$: Model size
* $N$: Number of clients
* $N_e$: Number of clients participating in federated evaluation per round

One $N$ for downloading model, one $N$ for uploading, and $N_e$ for evaluating.

Based on the raw training data of CIFAR-10, we have:
$$
B_{\rm baseline}=50000\times(32\times32\times3+1){\rm Byte}\approx146.53{\rm MB}
$$
where:

* $50000$ means there are $50000$ images
* $32\times32$ refers to the image size
* The explanation for $3$ is that under three color channels, each pixel can represent $(2^8)^3$ types of colors, where every 8-bit binary number is represented by one Byte.
* 1 is the label size that comes with each image

Finally:
$$
{\rm Bandwidth\ saved}=(1-\frac{B_{\rm FL}}{B_{\rm baseline}})\times100\%
$$

# Hierarchical FL (clients -> fog node → cloud)

Pipeline of standard FL:

> Client train -> Cloud aggregate client models through `FedAvg` -> Evaluate

For hierarchical FL

> Client train -> Fog node aggregate client models -> Cloud aggregate fog node models -> Evaluate

A possible solution is to modify `aggregate_train` of `FedAvg` to simulate fog nodes: Match clients and fog nodes, aggregate client models and send to fog nodes, then aggregate fog nodes.

Aggregation is done by weighted average.

## Bandwidth saved

Let:

* $R$: Training rounds
* $S$: Model size
* $N_c$: Number of clients
* $N_f$: Number of fog nodes
* $N_e$: Number of clients participating in federated evaluation per round

Bandwidth between clients and fog nodes:
$$
B_{\rm cli-fog}=RS(2N_c+N_e)
$$
Bandwidth between fog nodes and cloud:
$$
B_{\rm fog-cloud}=RS(2N_f)
$$
(Notice that evaluation does not happen in fog nodes in this case)

Hierarchical FL bandwidth:
$$
B_{\rm HFL}=B_{\rm cli-fog}+B_{\rm fog-cloud}=RS(2N_f+2N_c+N_e)
$$

# Test data

Test platform: Dell XPS 13 9315

CPU: Intel Core i5-1230U

GPU: Integrated

Memory: 16GB

Clients=10

Fog nodes=3

Local epochs=1

| Communication rounds | FL Type | Accuracy | Bandwidth saved | Training time |
| -------------------: | ------: | -------: | --------------: | ------------: |
|                    2 |     SFL |   11.85% |          90.26% |        39.80s |
|                    2 |     HFL |   11.76% |          88.31% |        38.18s |
|                    4 |     SFL |   16.67% |          80.52% |        60.10s |
|                    4 |     HFL |   18.92% |          76.63% |        62.46s |
|                    8 |     SFL |   27.53% |          61.04% |       110.73s |
|                    8 |     HFL |   26.74% |          53.25% |       110.72s |
|                   16 |     SFL |   31.53% |          22.09% |       201.51s |
|                   16 |     HFL |   27.94% |           6.50% |       203.82s |
|                   32 |     SFL |   35.69% |         -55.83% |       404.12s |
|                   32 |     HFL |   36.49% |         -86.99% |       391.09s |

# Discussion

Training is working, but the accuracy is low.

The bandwidth saved is acceptable under the current definition.

In this case, the accuracy and training time are consistent between SFL and HFL.

Overall, the data do not present a prominent distinction between SFL and HFL.

In the current implementation, the actual pipeline is

> Client -> Server -> Simulated fog nodes inside server

A possible solution is to implement real fog nodes. Further discussion needed.
