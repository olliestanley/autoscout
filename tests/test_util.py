from autoscout import util


def test_sleep_and_return():
    value = 1.011
    result = util.sleep_and_return(value, 0.01)
    assert value == result
