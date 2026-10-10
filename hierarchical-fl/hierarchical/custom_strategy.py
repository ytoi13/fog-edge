from flwr.app import ArrayRecord
from flwr.serverapp.strategy import FedAvg


class HierarchicalFedAvg(FedAvg):
    """Aggregate client models at fog nodes, then aggregate fog models at cloud."""

    def __init__(
        self, num_fog_nodes: int, fraction_evaluate: float
    ):
        self.num_fog_nodes = num_fog_nodes
        super().__init__(fraction_evaluate=fraction_evaluate)

    @staticmethod
    def aggregate_arrayrecords(records):
        """Perform weighted aggregation all ArrayRecords using a specific key."""
        total_examples = 0
        state_dicts = []
        for arrays, num_examples in records:
            total_examples += num_examples
            state_dicts.append(arrays.to_torch_state_dict())

        aggregated_arrays = {}

        for key in state_dicts[0]:
            weighted_sum = 0
            for state_dict, (_, num_examples) in zip(state_dicts, records):
                weighted_sum += state_dict[key] * num_examples
            aggregated_arrays[key] = weighted_sum / total_examples

        return ArrayRecord(aggregated_arrays), total_examples

    def aggregate_train(self, server_round, replies):
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        fog_records = []
        # Map clients to fog nodes using modulo mapping
        for fog_id in range(self.num_fog_nodes):
            client_records = []
            for msg in valid_replies:
                client_id = msg.metadata.src_node_id
                if client_id % self.num_fog_nodes == fog_id:
                    keys = msg.content[self.arrayrecord_key]
                    values = msg.content["metrics"][self.weighted_by_key]
                    client_records.append((keys, values))

            # Aggregate fog nodes ArrayRecords
            fog_records.append(self.aggregate_arrayrecords(client_records))

        reply_contents = [msg.content for msg in valid_replies]

        # Aggregate ArrayRecords
        arrays, _ = self.aggregate_arrayrecords(fog_records)

        # Aggregate MetricRecords
        metrics = self.train_metrics_aggr_fn(
            reply_contents,
            self.weighted_by_key
        )
        return arrays, metrics
