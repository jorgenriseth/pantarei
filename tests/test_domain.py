import dolfin as df
import pytest

from pantarei.mms import MMSInterval, MMSSquare


def test_mms_interval():
    domain = MMSInterval(10)
    dx = df.Measure("dx", domain=domain, subdomain_data=domain.subdomains)
    assert df.assemble(1 * dx(1)) == pytest.approx(2 * 0.8)
    assert df.assemble(1 * dx(2)) == pytest.approx(2 * (1 - 0.8))
    ds = df.Measure("ds", domain=domain, subdomain_data=domain.boundaries)
    for i in range(1, 2):
        assert df.assemble(1 * ds(i)) == pytest.approx(1)


def test_mms_square():
    domain = MMSSquare(20)
    dx = df.Measure("dx", domain=domain, subdomain_data=domain.subdomains)
    assert df.assemble(1 * dx(1)) == pytest.approx((2 * 0.8) ** 2)
    assert df.assemble(1 * dx(2)) == pytest.approx((2**2 - (2 * 0.8) ** 2))
    ds = df.Measure("ds", domain=domain, subdomain_data=domain.boundaries)
    for i in range(1, 5):
        assert df.assemble(1 * ds(i)) == pytest.approx(2)
