"""Testing forward propagation."""

import axon


def test_small_network():
  # Setup a basic network with no activations so we can test the outputs more
  # easily.
  network = axon.networks.Network()
  network.add_layer('input', 2)
  network.add_layer('output', 2, activation=None)
  # Override the weights and biases in the output layer.
  output_layer = network.layers[-1]
  output_layer.weights = [[3, 4],
                          [9, 8]]
  output_layer.biases = [[5],
                         [7]]
  # Forward propagate some data.
  vector = [10,
            20]
  network.forward_propagate(vector)
  # Check the output.
  expected_outputs = [
    [3 * 10 + 4 * 20 + 5],
    [9 * 10 + 8 * 20 + 7],
  ]
  assert output_layer.values == expected_outputs
