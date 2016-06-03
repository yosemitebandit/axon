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


one_hot_encodings = {
  'Iris-setosa': [1, 0, 0],
  'Iris-versicolor': [0, 1, 0],
  'Iris-virginica': [0, 0, 1],
}


# Setup input layer.
input_layer = axon.networks.Layer('input', 4)

# Setup the first hidden layer.
hidden_layer_one = axon.networks.Layer('hidden', 9)
hidden_layer_one.connect_to_parent(input_layer)

# Second the second hidden layer.
hidden_layer_two = axon.networks.Layer('hidden', 5)
hidden_layer_two.connect_to_parent(hidden_layer_one)

# Setup output layer.
output_layer = axon.networks.Layer('output', 3)
output_layer.connect_to_parent(hidden_layer_two)


print input_layer
print hidden_layer_one
print hidden_layer_two
print output_layer


axon.networks.forward_propagate(iris_data[0])
sys.exit()
estimate = axon.util.softmax([n.value for n in output_layer])
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
