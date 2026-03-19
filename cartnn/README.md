# About cartnn

This repository introduces small modifications to **e3nn**. Based on Cartesian-3j and Cartesian-nj, it implements **ICTP** and precomputed Cartesian product basis, thereby enabling Cartesian versions of **MACE**, **NequIP**, and **Allegro**.

**cartnn** is not recommended for practical use. The most important components of the code are:

```python
from cartnn.o3 import ICTD, cartesian_3j, CartesianHarmonics
```

# Install

The dependencies of this repository are the same as those of **e3nn**, and it currently does not support installation via `pip`.

```bash
git clone https://github.com/xvzemin/cartnn
cd cartnn/
pip install .
```

# Example 1
```python
        # === code === 
        import torch
        from cartnn.o3 import ICTD
        torch.set_printoptions(precision=4, sci_mode=False)

        batch = 1
        rank = 2

        gct = torch.randn(batch, *(3,)*rank) # generic Cartesian tensor
        print(gct)
        gct_flatten = gct.view(batch, -1)

        _, DS, _, _ = ICTD(rank) # obtain ictd matrix for each weight

        icts = []
        for D in DS:
            ict_flatten = gct_flatten @ D # irreducible Cartesian tensor
            ict = ict_flatten.view(batch, *(3,)*rank)
            print(ict)
            icts.append(ict)

        print(torch.allclose(gct, torch.stack(icts).sum(dim=0)))
```

# Example 2
```python
        # === code === 
        import torch
        from cartnn.o3 import ICTD
        torch.set_printoptions(precision=4, sci_mode=False)

        batch = 1
        rank = 2

        gct = torch.randn(batch, *(3,)*rank) # generic Cartesian tensor
        gct = gct.view(batch, -1)

        _, _, CS, SS = ICTD(rank) # obtain change-of-basis matrix

        for C, S in zip(CS, SS):
            st = gct @ C # Cartesian to spherical
            ict = st @ S # spherical to Cartesian
            print(st)
            print(ict.view(batch, *(3,)*rank))
```

# Example 3
```python
        # === code === 
        import torch
        from cartnn import o3
        torch.set_printoptions(precision=4, sci_mode=False)

        batch = 5
        max_ell = 3

        ch_irreps = o3.Irreps.cartesian_harmonics(max_ell, p=1)  # SO3
        ch_irreps = o3.Irreps.cartesian_harmonics(max_ell, p=-1) # O3
        cartesian_harmonics = o3.CartesianHarmonics(
            irreps_out=ch_irreps, 
            normalize=True, 
            norm=True, 
            traceless=True,
        )
        ch = cartesian_harmonics(torch.randn(batch, 3))

        print(ch.shape) # 1 + 3 + 9 + 27 = 40
```

# Citation

If you use Cartesian-nj in your work, we recommend citing both the cartnn-related paper and the original e3nn references.

```bash
@misc{xu2025cartesiannjextendinge3nnirreducible,
      title={Cartesian-nj: Extending e3nn to Irreducible Cartesian Tensor Product and Contracion}, 
      author={Zemin Xu and Chenyu Wu and Wenbo Xie and Daiqian Xie and P. Hu},
      year={2025},
      eprint={2512.16882},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2512.16882}, 
}
```