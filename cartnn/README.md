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

If you use cartnn, we recommend citing both the cartnn-related paper and the original e3nn references.

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

@misc{https://doi.org/10.48550/arxiv.2207.09453,
    doi = {10.48550/ARXIV.2207.09453},
    url = {https://arxiv.org/abs/2207.09453},
    author = {Geiger, Mario and Smidt, Tess},
    title = {e3nn: Euclidean Neural Networks},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}

@misc{thomas2018tensorfieldnetworks,
    title={Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds}, 
    author={Nathaniel Thomas and Tess Smidt and Steven Kearnes and Lusann Yang and Li Li and Kai Kohlhoff and Patrick Riley},
    year={2018},
    eprint={1802.08219},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/1802.08219}
}

@misc{weiler20183dsteerablecnns,
    title={3D Steerable CNNs: Learning Rotationally Equivariant Features in Volumetric Data}, 
    author={Maurice Weiler and Mario Geiger and Max Welling and Wouter Boomsma and Taco Cohen},
    year={2018},
    eprint={1807.02547},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/1807.02547}
}

@misc{kondor2018clebschgordannets,
    title={Clebsch-Gordan Nets: a Fully Fourier Space Spherical Convolutional Neural Network}, 
    author={Risi Kondor and Zhen Lin and Shubhendu Trivedi},
    year={2018},
    eprint={1806.09231},
    archivePrefix={arXiv},
    primaryClass={stat.ML},
    url={https://arxiv.org/abs/1806.09231}
}

@software{e3nn_software,
    author = {Mario Geiger and
              Tess Smidt and
              Alby M. and
              Benjamin Kurt Miller and
              Wouter Boomsma and
              Bradley Dice and
              Kostiantyn Lapchevskyi and
              Maurice Weiler and
              Michał Tyszkiewicz and
              Simon Batzner and
              Dylan Madisetti and
              Martin Uhrin and
              Jes Frellsen and
              Nuri Jung and
              Sophia Sanborn and
              Mingjian Wen and
              Josh Rackers and
              Marcel Rød and
              Michael Bailey},
    title = {Euclidean neural networks: e3nn},
    month = apr,
    year = 2022,
    publisher = {Zenodo},
    version = {0.5.0},
    doi = {10.5281/zenodo.6459381},
    url = {https://doi.org/10.5281/zenodo.6459381}
}
```