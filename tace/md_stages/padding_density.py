"""Opt4 padding validity is geometry, never a test of the smooth envelope."""
import math


def enable_opt4_density_masks_(model, cutoff):
    """Mask padding without dropping real edges with rounded-zero envelopes.

    Float32 polynomial envelopes can be exactly zero inside the cutoff. Those
    edges still contribute when native density cutoff is disabled. Opt2/Opt3
    retain their existing behavior; this setup is called only by Opt4.
    """
    if not math.isfinite(float(cutoff)) or cutoff <= 0:
        raise ValueError("Opt4 density padding cutoff must be finite and positive")
    anchor = next(model.parameters())
    patched = 0
    for module in model.modules():
        if not hasattr(module, "edge_density"):
            continue
        if not hasattr(module, "_opt4_density_padding_cutoff"):
            module.register_buffer(
                "_opt4_density_padding_cutoff", anchor.new_tensor(float(cutoff)),
                persistent=False,
            )
            patched += 1
    return patched


def apply_opt4_density_mask(density, cutoff, edge_length, padding_cutoff, apply_cutoff):
    """Keep the native smooth-envelope derivative when the checkpoint uses it."""
    if cutoff is not None and apply_cutoff:
        density = density * cutoff
    return density * (edge_length < padding_cutoff).to(dtype=density.dtype)
