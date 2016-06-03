"""Layer and Network classes."""

import math
import random

import axon


class Network(object):
  def __init__(self):
    self.layers = []

  def __repr__(self):
    return '%s-layer network' % len(self.layers)

  def add_layer(self, kind, size):
    new_layer = Layer(kind, size)
    if len(self.layers) > 0:
      parent_layer = self.layers[-1]
      new_layer.set_weights_based_on_parent(parent_layer)
    self.layers.append(new_layer)

  def forward_propagate(self, data):
    """Send one vector of data through the network."""
    # Set the input layer values.
    input_layer = self.layers[0]
    for i, value in enumerate(data):
      input_layer.values[i][0] = value
    for index, layer in enumerate(self.layers):
      if index == 0:
        continue
      # Wx + b
      self.layers[index].values = (
        axon.util.col_vector_sum(
          axon.util.matrix_product(self.layers[index].weights,
                                   self.layers[index - 1].values),
          self.layers[index].biases))

  def make_estimate(self):
    """Get the latest estimate via softmax."""
    output_layer_values = axon.util.flatten(self.layers[-1].values)
    return axon.util.softmax(output_layer_values)

  def back_propagate(self, actual_encoding, learning_rate=0.1):
    """Run the back prop algorithm to update weights."""
    # Calculate the delta values for each layer.
    reversed_layers = [l for l in reversed(self.layers)]
    for layer_index, layer in enumerate(reversed_layers):
      if layer.kind == 'output':
        estimated_encoding = self.make_estimate()
        for index, estimated_value in enumerate(estimated_encoding):
          layer.deltas[index] = (
            (estimated_value - actual_encoding[index]) *
            estimated_value * (1 - estimated_value))
      elif layer.kind == 'hidden':
        for value_index, value in enumerate(layer.values):
          child_layer_weights = [
            r[value_index] for r in reversed_layers[layer_index-1].weights]
          child_layer_errors_sum = 0
          for i, child_weight in enumerate(child_layer_weights):
            child_layer_errors_sum += (
              child_weight * reversed_layers[layer_index-1].deltas[i])
          layer.deltas[value_index] = (
            value[0] * (1 - value[0]) * child_layer_errors_sum)
    # Now use the deltas to update weights and biases in each layer.
    for layer_index, layer in enumerate(self.layers):
      if layer.kind == 'input':
        continue
      for weight_row_index, weight_row in enumerate(layer.weights):
        for weight_value_index in range(len(weight_row)):
          weight_row[index] += (
            -1 * learning_rate * layer.deltas[weight_row_index] *
            self.layers[layer_index-1].values[weight_value_index][0])
        layer.biases[weight_row_index][0] += (
          learning_rate * layer.deltas[weight_row_index])



class Layer(object):
  def __init__(self, kind, size):
    self.kind = kind
    # Want a size x 1 sized col vector.
    self.values = [[0] for _ in range(size)]
    self.weights = []
    self.biases = [[random.random() * 2 - 1] for _ in range(size)]
    self.deltas = [0 for _ in range(size)]

  def __repr__(self):
    if self.weights:
      weight_rows = len(self.weights)
      weight_cols = len(self.weights[0])
      return '%s layer, %s nodes, %s x %s weights' % (
        self.kind, len(self.values), weight_rows, weight_cols)
    else:
      return '%s layer, %s nodes' % (self.kind, len(self.values))

  def set_weights_based_on_parent(self, parent_layer):
    # Init weights -- each inner array is a row..
    # Parent layer defines the cols.
    self.weights = [
      [random.random() * 2 - 1 for _ in parent_layer.values]
      for _ in self.values]


class Node(object):
  def __init__(self, kind='hidden'):
    self.kind = kind
    self.value = None
    self.parent_nodes = []
    self.child_nodes = []
    self.parent_weights = []
    self.parent_weight_deltas = []
    self.child_weights = []
    self.bias = random.random() * 2 - 1
    self.delta = None

  def set_parent_with_weight(self, parent_node, weight):
    """Create a weighted connection to a parent node."""
    self.parent_nodes.append(parent_node)
    self.parent_weights.append(weight)
    # Also tell the parent that it has an attached child node.
    parent_node.child_nodes.append(self)
    parent_node.child_weights.append(weight)

  def forward_propagate(self):
    """Update the node's value based on the values of the parents."""
    self.value = 0
    for i, parent_node in enumerate(self.parent_nodes):
      self.value += self.parent_weights[i] * parent_node.value
    self.value += self.bias
    # ReLU activation.
    # self.value = max(0, self.value)
    # Sigmoid activation.
    self.value = 1 / (1 + math.exp(-self.value))

  def back_propagate(self, target_value=0, learning_rate=0.1):
    """Update weights based on target values.

    Specify a target_value if this is an output node.
    """
    # Calculate delta based on kind.
    if self.kind == 'output':
      self.delta = self.value * (1 - self.value) * (self.value - target_value)
    elif self.kind == 'hidden':
      child_errors_sum = 0
      for i, child_node in enumerate(self.child_nodes):
        child_errors_sum += child_node.delta * self.child_weights[i]
      self.delta = self.value * (1 - self.value) * child_errors_sum
    # Get parent weight deltas.
    for i, parent_node in enumerate(self.parent_nodes):
      self.parent_weight_deltas[i] = (
        -1 * learning_rate * self.delta * parent_node.value)
    # Separately update weights.
    for i, parent_node in enumerate(self.parent_nodes):
      self.parent_weights[i] += self.parent_weight_deltas[i]
    # Update bias term.
    self.bias += learning_rate * self.delta
