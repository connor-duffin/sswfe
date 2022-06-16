import pytest

from swe import ShallowTwo


@pytest.fixture
def swe():
    control = {"nx": 32, "dt": 0.02, "theta": 1}
    return ShallowTwo(control)


def test_shallowtwo_init(swe):
    assert swe.nx == 32
    assert swe.dt == 0.02
    assert swe.theta == 1
