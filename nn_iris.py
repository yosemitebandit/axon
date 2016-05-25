"""Network to classify the iris dataset."""


import math


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


class Node(object):
  def __init__(self, value):
    self.value = value
    self.parents = []
    self.weights = []

  def __repr__(self):
    return 'Node(%s), %s parents, %s weights' % (
      self.value, len(self.parents), len(self.weights))

  def connect_to_parent_with_weight(self, node, weight):
    """Create a weighted connection to a parent node."""
    self.parents.append(node)
    self.weights.append(weight)

  def forward_propagate(self):
    """Update the node's value based on the values of the parents."""
    self.value = 0
    for i, parent_node in enumerate(self.parents):
      self.value += self.weights[i] * parent_node.value


# Setup input layer.
input_nodes = 4
input_layer = []
for _ in range(input_nodes):
  input_layer.append(Node(None))


# Setup a single hidden layer.
hidden_nodes = 4
hidden_layer = []
for _ in range(hidden_nodes):
  hidden_node = Node(None)
  average_weight = 1. / hidden_nodes
  for input_node in input_layer:
    hidden_node.connect_to_parent_with_weight(input_node, average_weight)
  hidden_layer.append(hidden_node)

# Setup the output layer.
output_nodes = 3
output_layer = []
for _ in range(output_nodes):
  output_node = Node(None)
  average_weight = 1. / output_nodes
  for hidden_node in hidden_layer:
    output_node.connect_to_parent_with_weight(hidden_node, average_weight)
  output_layer.append(output_node)


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


forward_propagate(iris_data[3])
print output_layer[2]

print softmax([n.value for n in output_layer])
