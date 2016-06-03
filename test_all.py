import axon


def test_col_vector_sum():
  a = [[0], [1], [2]]
  b = [[3], [4], [5]]
  expected = [[3], [5], [7]]
  assert axon.util.col_vector_sum(a, b) == expected
