import torch
import torch.nn as nn

torch.manual_seed(0)


num_experts = 16
num_shared_experts = 8
top_k = 4
dim = 1000000  

weight2 = torch.randn(num_experts, dim) 
weight1_all = torch.randn(num_shared_experts, dim)
alpha = 1.0 / (num_shared_experts ** 0.5)
weight_shared = alpha * weight1_all.sum(dim=0)


logits = torch.randn(num_experts)
probs = torch.softmax(logits, dim=0)

topk_probs, topk_idx = torch.topk(probs, k=top_k)
gate = torch.zeros_like(probs)
gate[topk_idx] = topk_probs / topk_probs.sum()


weight_router = torch.einsum("z, zi -> i", gate, weight2)

router_l2 = (gate ** 2).sum()

weight_raw = weight_shared + weight_router


scale = torch.sqrt(1.0 + router_l2)
weight_scaled = weight_raw / scale


def stat(x, name):
    print(f"{name:20s} mean={x.mean().item():+.4f}, var={x.var(unbiased=False).item():.4f}")


stat(weight_shared, "shared (combined)")
stat(weight_router, "router")
print(f"router L2 = {router_l2.item():.4f}")
print(f"expected raw var ≈ {1.0 + router_l2.item():.4f}")
stat(weight_raw, "raw total")
stat(weight_scaled, "scaled total")