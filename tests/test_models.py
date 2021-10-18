"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest
from unittest.mock import patch


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""
    from inflammation.models import daily_mean

    # NB: the comment 'yapf: disable' disables automatic formatting using
    # a tool called 'yapf' which we have used when creating this project
    test_array = np.array([[0, 0],
                           [0, 0],
                           [0, 0]])  # yapf: disable

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array([0, 0]), daily_mean(test_array))


def test_daily_mean_integers():
    """Test that mean function works for an array of positive integers."""
    from inflammation.models import daily_mean

    test_array = np.array([[1, 2],
                           [3, 4],
                           [5, 6]])  # yapf: disable

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(np.array([3, 4]), daily_mean(test_array))


def test_daily_max_integers():
    """Tests max function works in an array"""

    from inflammation.models import daily_max

    test_array = np.array([[0,2],[9,3],[1,8],[2,2]]) #yapf: disable
    npt.assert_array_equal(np.array([9, 8]), daily_max(test_array))

@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0,1,7], [6,5,8], [5,3,0]], [0,1,0]),
        ([[7,1,8], [3,4,1], [1,9,3]], [1,1,1]),
    ])
def test_daily_min_integers(test,expected):
    """Tests max function works in an array"""

    from inflammation.models import daily_min

    #test_array = np.array([[0,2,1],[9,3,7],[1,8,3],[2,2,6]])
    npt.assert_array_equal(np.array(expected), daily_min(test))

@patch('inflammation.models.get_data_dir', return_value='/data_dir')
def test_load_csv(mock_get_data_dir):
    from inflammation.models import load_csv
    with patch('numpy.loadtxt') as mock_loadtxt:
        load_csv('test.csv')
        name, args, kwargs = mock_loadtxt.mock_calls[0]
        assert kwargs['fname'] == '/data_dir/test.csv'
        load_csv('/test.csv')
        name, args, kwargs = mock_loadtxt.mock_calls[1]
        assert kwargs['fname'] == '/test.csv'

# TODO(lesson-automatic) Implement tests for the other statistical functions
# TODO(lesson-mocking) Implement a unit test for the load_csv function
