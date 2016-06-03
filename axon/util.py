"""Util functions."""

import math


def col_vector_sum(a, b):
  """Sum two col vectors."""
  result = [[0] for _ in a]
  for i, element in enumerate(a):
    result[i][0] = element[0] + b[i][0]
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
