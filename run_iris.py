"""Classifying the iris dataset."""

import random

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
network.add_layer('hidden', 9, activation='sigmoid')
network.add_layer('hidden', 5, activation='sigmoid')
network.add_layer('output', 3, activation='sigmoid')
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
estimated_encoding = network.make_estimate()
actual_encoding = one_hot_encodings[iris_data[0]['name']]
output_error = axon.util.mean_squared_error_sum(
  estimated_encoding, actual_encoding)
print estimated_encoding
print actual_encoding
print output_error


# Back propagate errors to update weights based on the actual encoding.
network.back_propagate(actual_encoding)


# Try the estimation again and compare error rates.
network.forward_propagate(vector)
estimated_encoding = network.make_estimate()
actual_encoding = one_hot_encodings[iris_data[0]['name']]
output_error2 = axon.util.mean_squared_error_sum(
  estimated_encoding, actual_encoding)
print '\n\n'
print estimated_encoding
print actual_encoding
print output_error2


# Now run it a lot and chart the errors.
errors = []
for _ in range(100):
  random.shuffle(iris_data)
  for data in iris_data:
    vector = [
      data['sepal_length'],
      data['sepal_width'],
      data['petal_length'],
      data['petal_width'],
    ]
    network.forward_propagate(vector)
    estimated_encoding = network.make_estimate()
    actual_encoding = one_hot_encodings[data['name']]
    output_error = axon.util.mean_squared_error_sum(
      estimated_encoding, actual_encoding)
    network.back_propagate(actual_encoding, learning_rate=0.01)
  errors.append(output_error)

print 'oe', output_error

import matplotlib.pyplot as plt
plt.plot(errors)
plt.show()
