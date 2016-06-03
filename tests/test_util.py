"""Testing the axon.util module."""

import pytest

import axon


def test_col_vector_sum():
  a = [[0],
       [1],
       [2]]
  b = [[3],
       [4],
       [5]]
  expected = [[3],
              [5],
              [7]]
  assert axon.util.col_vector_sum(a, b) == expected


def test_matrix_product_dim_check():
  a = [[0],
       [1],
       [2]]
  b = [[3],
       [4]]
  with pytest.raises(TypeError):
    axon.util.matrix_product(a, b)


def test_matrix_product_one_by_one():
  a = [[3]]
  b = [[2]]
  expected = [[6]]
  assert axon.util.matrix_product(a, b) == expected


def test_matrix_product_larger():
  a = [[0, 1, 2],
       [2, 3, 2]]
  b = [[2, 3, 4, 0],
       [1, 2, 3, 0],
       [2, 4, 6, 0]]
  expected = [[5,  10, 15, 0],
              [11, 20, 29, 0]]
  assert axon.util.matrix_product(a, b) == expected


def test_flatten():
  a = [[0],
       [1],
       [2]]
  expected = [0, 1, 2]
  assert axon.util.flatten(a) == expected
