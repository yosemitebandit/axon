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


def softmax(values):
  """Compute the softmax."""
  return [math.exp(v) / sum([math.exp(v) for v in values]) for v in values]


def mean_squared_error(v1, v2):
  """The element-wise mean squared error between two vectors."""
  mse = []
  for i, v in enumerate(v1):
    mse.append(0.5 * (v - v2[i]) ** 2)
  return mse
