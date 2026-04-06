import torch
import math

torch.set_default_dtype(torch.float64)
class jnTaylorSphericalBessel(torch.nn.Module):
    def __init__(self, n, K=6):
        super().__init__()
        self.n = n
        self.K = K
        prefactor = []
        for k in range(self.K):
            prefactor.append(
                ((-1)**k)
                / (math.factorial(k) * math.gamma(k + n + 1.5))
                * 0.5 * math.sqrt(math.pi) 
            )
        self.register_buffer('prefactor', torch.tensor(prefactor))
        self.register_buffer('powers', 2*torch.arange(self.K) + self.n, persistent=False)

    def forward(self, x: torch.Tensor)  -> torch.Tensor:
        orig_dtype = x.dtype
        orig_shape = x.shape
        x = x.to(torch.float64)
        x = x.view(-1)
        x = 0.5 * x
        x_pow = x.unsqueeze(-1) ** self.powers  
        prefactor = self.prefactor.to(x.dtype)
        return torch.sum(prefactor * x_pow, dim=-1).to(orig_dtype).reshape(orig_shape)
    

def j0_exact(x):
    return torch.where(x == 0, torch.tensor(1.0, device=x.device), torch.sin(x)/x)

def j1_exact(x):
    return torch.where(
        x == 0,
        torch.tensor(0.0, device=x.device),
        torch.sin(x)/x**2 - torch.cos(x)/x
    )

def j2_exact(x):
    return torch.where(
        x == 0,
        torch.tensor(0.0, device=x.device),
        (3/x**3 - 1/x)*torch.sin(x) - 3*torch.cos(x)/x**2
    )

x = torch.tensor([0.01, 0.1, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)

models = [
    (0, jnTaylorSphericalBessel(0)),
    (1, jnTaylorSphericalBessel(1)),
    (2, jnTaylorSphericalBessel(2)),
]

exact_fns = [j0_exact, j1_exact, j2_exact]

for (n, model), exact_fn in zip(models, exact_fns):
    print(f"\n===== n = {n} =====")

    approx = model(x)
    exact = exact_fn(x)

    for i in range(len(x)):
        print(f"x = {x[i].item():.2f}")
        print(f"approx = {approx[i].item():.12f}")
        print(f"exact  = {exact[i].item():.12f}")
        print(f"error  = {abs(approx[i]-exact[i]).item():.2e}")
        print()