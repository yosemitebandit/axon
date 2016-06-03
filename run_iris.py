"""Classifying the iris dataset."""

import sys

import axon


# Extract iris data.
iris_data = []
with open('iris.csv') as iris_file:
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


# Setup encodings.
one_hot_encodings = {
  'Iris-setosa': [1, 0, 0],
  'Iris-versicolor': [0, 1, 0],
  'Iris-virginica': [0, 0, 1],
}


# Setup the network.
network = axon.networks.Network()
network.add_layer('input', 4)
network.add_layer('hidden', 9)
network.add_layer('hidden', 5)
network.add_layer('output', 3)
print network
for layer in network.layers:
  print layer


# Forward propagate one line of the CSV data.
vector = [
  iris_data[0]['sepal_length'],
  iris_data[0]['sepal_width'],
  iris_data[0]['petal_length'],
  iris_data[0]['petal_width'],
]
network.forward_propagate(vector)
print network.estimate()
sys.exit()
actual = one_hot_encodings[iris_data[0]['name']]
output_error = sum(axon.util.mean_squared_error(estimate, actual))


'''
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
'''
