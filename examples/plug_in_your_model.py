"""Minimal example: wrap your affinity model and run FamBench.

A model is any callable ``predict(sequence, smiles) -> float`` where a higher return
value means a stronger binder. If your model outputs log10(IC50) or similar
(lower = stronger), pass ``higher_is_stronger=False`` to the run_* functions.
"""
import fambench


# ---------------------------------------------------------------------------
# 1. Wrap your model. Replace the body with a real call to your predictor.
#    (Here we use the molecular-weight baseline as a stand-in.)
# ---------------------------------------------------------------------------
my_model = fambench.baseline_model("molecular_weight")

# A real example would look like:
#
#   from my_package import load_model
#   _net = load_model("checkpoint.pt")
#   def my_model(sequence: str, smiles: str) -> float:
#       return _net.predict_affinity(sequence, smiles)   # higher = stronger


# ---------------------------------------------------------------------------
# 2. Run the two arms.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Redundancy arm: how does accuracy scale with protein-family representation?
    red = fambench.run_redundancy(my_model, core=True)   # core = balanced 3,360 subset
    print(red.summary())
    print()

    # Temporal arm: a genuinely novel target. Do you beat molecular weight?
    tmp = fambench.run_temporal(my_model)
    print(tmp.summary())

    # The headline numbers you should report:
    print("\n--- headline ---")
    print(f"novel-family Pearson : {red.novel_family_pearson:+.3f}")
    print(f"leakage slope (t)    : {red.leakage_slope:+.3f} ({red.leakage_slope_t:+.1f})")
    print(f"novel-target Spearman: {tmp.spearman:+.3f}  "
          f"(molecular weight: {tmp.baseline_spearman.get('molecular_weight'):+.3f})")
