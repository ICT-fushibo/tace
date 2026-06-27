import torch
from tace.models.so2.wigner import WignerD

mmax = 2
lmax = 4

dtype = torch.float64
atol = 1e-5
rtol = 1e-5

def test_cartesian_get_wigner_matches_euler_with_truncated_m() -> None:
    edge_vector = torch.randn(8, 3, dtype=dtype)
    cartesian = WignerD(lmax, mmax, wigner_type="flash")
    xuzemin = WignerD(lmax, mmax, wigner_type="ictd")
    euler = WignerD(lmax, mmax, wigner_type="euler")
    torch.manual_seed(7)
    actual, actual_inv = cartesian.get_wigner(edge_vector)
    torch.manual_seed(7)
    expected, expected_inv = euler.get_wigner(edge_vector)
    torch.manual_seed(7)
    xzm, xzm_inv = xuzemin.get_wigner(edge_vector)
    torch.testing.assert_close(xzm, expected, atol=atol, rtol=rtol)

if __name__ == "__main__":
    test_cartesian_get_wigner_matches_euler_with_truncated_m()
    print("Pass")