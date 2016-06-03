"""Network to classify the iris dataset."""


import math
import random
import sys

import matplotlib.pyplot as plt


# Extract iris data.
iris_data = []
with open('iris.txt') as iris_file:
  for line in iris_file.read().split('\n'):
    if not line:
      continue
    sepal_length, sepal_width, petal_length, petal_width, name = (
      line.split(','))
    iris_data.append({
      'name': name,
      'sepal_length': float(sepal_length),
      'sepal_width': float(sepal_width),
      'petal_length': float(petal_length),
      'petal_width': float(petal_width),
    })


one_hot_encodings = {
  'Iris-setosa': [1, 0, 0],
  'Iris-versicolor': [0, 1, 0],
  'Iris-virginica': [0, 0, 1],
}


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


# Setup input layer.
input_layer = Layer('input', 4)

# Setup the first hidden layer.
hidden_layer_one = Layer('hidden', 9)
hidden_layer_one.connect_to_parent(input_layer)

# Second the second hidden layer.
hidden_layer_two = Layer('hidden', 5)
hidden_layer_two.connect_to_parent(hidden_layer_one)

# Setup output layer.
output_layer = Layer('output', 3)
output_layer.connect_to_parent(hidden_layer_two)


print input_layer
print hidden_layer_one
print hidden_layer_two
print output_layer
sys.exit()


# Setup input layer.
# input_nodes = 4
# input_layer = []
# for _ in range(input_nodes):
  # input_layer.append(Node(kind='input'))

# Setup a single hidden layer.
# hidden_nodes = 4
# hidden_layer = []
# for _ in range(hidden_nodes):
  # hidden_node = Node(kind='hidden')
  # for input_node in input_layer:
    # weight = 2*random.random() - 1
    # hidden_node.set_parent_with_weight(input_node, weight)
  # hidden_layer.append(hidden_node)


# Setup the output layer.
# output_nodes = 3
# output_layer = []
# for _ in range(output_nodes):
  # output_node = Node(kind='output')
  # for hidden_node in hidden_layer:
    # weight = 2*random.random() - 1
    # output_node.set_parent_with_weight(hidden_node, weight)
  # output_layer.append(output_node)


def forward_propagate(data):
  """Send one line of iris data through the network.

  Args:
    data: a dict with keys sepal_length, sepal_width, petal_length, petal_width
          and name
  """
  n1, n2, n3, n4 = input_layer
  n1.value = data['sepal_length']
  n2.value = data['sepal_width']
  n3.value = data['petal_length']
  n4.value = data['petal_width']
  # Update the hidden layer.
  for node in hidden_layer:
    node.forward_propagate()
  # Update output layer.
  for node in output_layer:
    node.forward_propagate()


def softmax(values):
  """Compute the softmax."""
  return [math.exp(v) / sum([math.exp(v) for v in values]) for v in values]


def mean_squared_error(v1, v2):
  """The element-wise mean squared error between two vectors."""
  mse = []
  for i, v in enumerate(v1):
    mse.append(0.5 * (v - v2[i]) ** 2)
  return mse


forward_propagate(iris_data[0])
estimate = softmax([n.value for n in output_layer])
actual = one_hot_encodings[iris_data[0]['name']]
output_error = sum(mean_squared_error(estimate, actual))


def back_propagate(target_encoding):
  """Back propagate errors through the layers.

  Args:
    a ground-truth one-hot encoding

  Returns:
    None, but this will update node weights
  """
  # Update the output layer.
  for i, node in enumerate(output_layer):
    node.back_propagate(target_value=target_encoding[i])
  # Update the hidden layer.
  for node in hidden_layer:
    node.back_propagate()


errors = []
for iteration in range(50):
  for data in iris_data:
    forward_propagate(data)
    estimate = softmax([n.value for n in output_layer])
    actual = one_hot_encodings[data['name']]
    output_error = sum(mean_squared_error(estimate, actual))
    back_propagate(actual)
    errors.append(output_error)


print output_error
plt.plot(errors)
plt.show()
