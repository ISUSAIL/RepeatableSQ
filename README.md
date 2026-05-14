# Rethink Repeatable Measures of Robot Performance with Statistical Query

## Citation

If you use this work, please cite:

```bibtex
@article{weng2024rethink,
  title={Rethink Repeatable Measures of Robot Performance with Statistical Query},
  author={Weng, Bowen and Capito, Linda and Castillo, Guillermo A. and Khor, Dylan},
  journal={IEEE Transactions on Robotics},
  year={2025},
  publisher={IEEE}
}
```

## Repository Structure

### Root Directory Files
- `stat_query.py` - A replication of the paper's primary algorithms:
- `distributions.py` - Distribution utilities used throughout the project

### Directories
- `demo/` - Contains synthetic distributions and demonstrations:
  - `01-demo-distribution.ipynb` - Basic distribution demonstrations
  - `02-demo-stat-query.ipynb` - Statistical query demonstrations
  - `03-demo-repeatable-stat-query.ipynb` - Repeatable statistical query demonstrations
 
- `images/` - Visualization outputs and figures

## Correction Note

`T_RO_RethinkRepeatability_correction.pdf` documents a corrigendum to the published article. The integral evaluated in equation (12) of the paper is incorrect, and the error propagates into equations (13)–(15) (Theorem 2) and the validity condition in Remark 4. The corrected closed form is

$$\alpha \\;=\\; 3\gamma\\,\frac{(1-c) - \sqrt{(1-c)^2 - \tfrac{4}{3}(1-\beta)}}{1-c},
\quad\text{valid when } (1-c)^2 \geq \tfrac{4}{3}(1-\beta),$$

replacing the original constant `1` on the right-hand side of the validity condition with `4/3`, and rescaling the closed form by a factor of `3` with `4/3` inside the square root. All qualitative conclusions, definitions, the `γ + α/2`-accuracy result, the Lyapunov stability argument, and the empirical disagreement frequencies reported in the article are unaffected — only the closed-form constants in the accuracy/repeatability trade-off change. See the PDF for the full derivation and the structural assumption (Assumption 1) made explicit for the CDF dominance step.

**Code is intentionally left unchanged** so the repository continues to reproduce the formulas as printed in the original IEEE T-RO publication. To use the corrected relation, swap in the coefficients above where the original equation (15) is computed — e.g., replace `γ` with `3γ` as the outer factor and replace `(1 − β)` with `(4/3)(1 − β)` inside the square root. The accuracy bound `γ + α/2` and all sampling/estimation code remain valid as-is for any chosen `α`.