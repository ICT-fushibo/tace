import torch
from e3nn import o3

# =========================
# 1. build model
# =========================
irreps = "64x0e + 64x1o + 64x2e + 64x3o + 64x4e + 64x5o"

linear = o3.Linear(irreps, irreps)

device = "cuda" if torch.cuda.is_available() else "cpu"
linear = linear.to(device)

torch.manual_seed(0)

x = torch.randn(4, linear.irreps_in.dim, device=device)
x_ref = x.clone()


def set_identity(linear, scale=1.0):
    with torch.no_grad():
        linear.weight.zero_()
        eye = torch.eye(64, device=linear.weight.device)
        for wv in linear.weight_views():
            wv.copy_(scale * eye)


def test_scale(scale):
    set_identity(linear, scale=scale)
    y = linear(x)

    err = (y - x_ref).abs().mean().item()
    max_err = (y - x_ref).abs().max().item()

    return err, max_err


scales = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

results = []

for s in scales:
    err, max_err = test_scale(s)
    results.append((s, err, max_err))
    print(f"scale={s:6.3f}  mean_err={err:.6e}  max_err={max_err:.6e}")


best = min(results, key=lambda x: x[1])

print("\nBEST SCALE:")
print(best)


set_identity(linear, scale=best[0])
y = linear(x)

print("\nFINAL CHECK:")
print("mean error:", (y - x_ref).abs().mean().item())
print("max error :", (y - x_ref).abs().max().item())

assert torch.allclose(y, x_ref, atol=1e-4), "NOT identity"
print("✔ identity verified")