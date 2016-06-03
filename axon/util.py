"""Util functions."""

import math


def col_vector_sum(a, b):
  """Sum two col vectors."""
  result = [[0] for _ in a]
  for i, element in enumerate(a):
    result[i][0] = element[0] + b[i][0]
  return result


def matrix_product(a, b):
  """Calculate the product between 2D matrices."""
  # Dimension check: len of a's cols must equal len of b's rows.
  if len(a[0]) != len(b):
    raise TypeError
  # Allocate.
  result = [[0 for _ in b[0]] for _ in a]
  # Compute the dot product to find the values in the result matrix.
  for row_index in range(len(result)):
    for col_index in range(len(result[0])):
      a_values = a[row_index]
      b_values = [row[col_index] for row in b]
      dot_product = 0
      for i, a_value in enumerate(a_values):
        dot_product += a_value * b_values[i]
      result[row_index][col_index] = dot_product
  return result


def flatten(column_matrix):
  """Flattens a column matrix into a vector."""
  return [v[0] for v in column_matrix]


def softmax(values):
  """Compute the softmax."""
  return [math.exp(v) / sum([math.exp(v) for v in values]) for v in values]


def mean_squared_error_sum(a, b):
  """The element-wise mean squared error between two vectors."""
  errors = []
  for index, value in enumerate(a):
    errors.append(0.5 * (value - b[index]) ** 2)
  return sum(errors)


def sigmoid(x):
  """Computes the sigmoid for a value, x."""
  return 1 / (1 + math.exp(-1 * x))
