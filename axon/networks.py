"""Layer and Network classes."""

import math
import random

import axon


class Layer(object):
  def __init__(self, kind, size):
    self.kind = kind
    # Want a size x 1 sized col vector.
    self.values = [[0] for _ in range(size)]
    self.parent_layer = None
    self.child_layer = None
    self.weights = []
    self.biases = [[random.random() * 2 - 1] for _ in range(size)]
    self.deltas = []

  def __repr__(self):
    if self.weights:
      weight_rows = len(self.weights)
      weight_cols = len(self.weights[0])
      return '%s layer, %s nodes, %s x %s weights' % (
        self.kind, len(self.values), weight_rows, weight_cols)
    else:
      return '%s layer, %s nodes' % (self.kind, len(self.values))

  def connect_to_parent(self, parent_layer):
    self.parent_layer = parent_layer
    # Init weights -- each inner array is a row..
    # Parent layer defines the cols.
    self.weights = [
      [0 for _ in parent_layer.values] for _ in self.values]
    # Setup child layer as well.
    parent_layer.child_layer = self


def forward_propagate(input_layer, data):
  """Send one line of iris data through the network.

  Args:
    input_layer: the first layer of the network
    data: a dict with keys sepal_length, sepal_width, petal_length, petal_width
          and name
  """
  input_layer.values = [
    data['sepal_length'],
    data['sepal_width'],
    data['petal_length'],
    data['petal_width'],
  ]
  # Update subsequent layers.
  current_layer = input_layer
  next_layer = input_layer.parent_layer
  while True:
    print current_layer, next_layer
    # Wx + b
    next_layer.values = (
      axon.util.col_vector_sum(
        axon.util.dot_product(next_layer.weights, current_layer.values),
        next_layer.biases))
    if not next_layer.parent:
      break
    else:
      current_layer = next_layer
      next_layer = next_layer.parent


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
