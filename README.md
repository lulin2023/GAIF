# GAIF Code Overview

This repository contains the code to reproduce the experiments results in **Feedback-Enhanced Online Multiple Testing with Applications to Conformal Selection** by Lin Lu, Yuyang Huo, Haojie Ren, Zhaojun Wang and Changliang Zou.


## Folder contents

- `1-useful functions/`: contains the R functions to implement all proposed methods (**SF, LF, SFS, LFS, OCTF and Opt-OCTF**).
- `2-Synthetic-data/`: contains the code for synthetic data simulation experiments in the paper.
- `3-Real-data/`: contains the dataset folders and code for real data applications.

---

## Guide for the codes in `1-useful functions/` folder

This folder contains the underlying algorithmic definitions, helper functions, and feedback strategies:
- `algoclass_OnSel.R`
- `functions_OnSel.R`
- `local dependence func.R`
- `Model-sel-func.R`
- `SAFFRON_feedback functions.R`

---

## Guide for the codes in `2-Synthetic-data/` folder

Code for reproducing results in Section 4, Appendix E, and various figures/tables in the paper.

### 1-non-conformal setting
- `0-Illustration_figure(Figure1).R`
- `1-Simulation_GAIF_Gaussian(Figure2-I).R`
- `2-Simulation_GAIF_Beta(Figure2-II).R`
- `3-Simulation_GAIF_dep(Figure3).R`
- `4-Simulation_GAIF_Gaussian_Bandit(Figure4).R`
- `5-Simulation_GAIF_Gaussian_Delayed(Figure5).R`
- `6-plots_mFDR(Figures19-20).R`

### 2-conformal testing
- `1-simu_cla_OCTF(Figure6).R`
- `2-simu_cla_Opt_OCTF_shift_online(Figure7).R`
- `3-simu_cla_OCTF-delayed(Figure10).R`
- `4-simu_cla_Opt_online_updating(Table2).R`
- `5-simu_cla_null_drift(Figures11-12).R`
- `6-airfoil_covshift(Figure13).R`
- `7-simu_reg_OCTF(Figure14).R`
- `8-simu_cla_OCTF_different_ncal(Figure15).R`
- `9-simu_reg_OCTF_different_ncal(Figure16).R`
- `10-simu_cla_Opt_OCTF(Figure17).R`
- `11-simu_reg_Opt_OCTF(Figure18).R`
- `12-Opt-SFS-variant(Figure21).R`
- `13-Figure 22.R`

---

## Guide for the codes in `3-Real-data/` folder
Data and code folders for reproducing real-world applications in Section 5.

- `airfoil/` (Task: Airfoil noise detection )
- `candidate/` (Task: Job candidate selection/screening)
- `diabetes/` (Task: Diabetes clinical indicators analysis)
- `income/` (Task: Census income level prediction)

---

## Citation
If you find this work useful, you can cite it with the following BibTex entry:

```bibtex
@misc{lu2025feedbackenhancedonlinemultiple,
      title={Feedback-Enhanced Online Multiple Testing with Applications to Conformal Selection},
      author={Lin Lu and Changliang Zou and Zhaojun Wang and Haojie Ren and Yuyang Huo},
      year={2025},
      eprint={2509.03297},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={[https://arxiv.org/abs/2509.03297](https://arxiv.org/abs/2509.03297)},
}
