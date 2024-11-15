# Detecting Human Settlements

The competition details can be reviewed at [INEGI-GCIM Human Settlement Detection Challenge](https://zindi.africa/competitions/inegi-gcim-human-settlement-detection-challenge).

Using the code provided in this GitHub repository, I achieved the following scores:

| Place | Leaderboard        | Public   | Private   |
|-------|--------------------|----------|-----------|
| 1     | Click Click Boom   | 0.8614   | 0.8627    |
|       | **S5**             | **0.8628** | **0.8662** |

---

# Approach 

The solution implements the **Simplified Structured State Space Sequencing (S5)** model, as described by [Smith et al.](https://arxiv.org/abs/2208.04933). The S5 model extends the S4 ideas of [Gu et al.](https://arxiv.org/abs/2111.00396) by using multi-input, multi-output state space models. In turn, S4 expands on the concept of linear time-invariant dynamical systems (LTI), which are governed by the following continuous equations:

$$
\dot{\mathbf{x}}(t) = \mathbf{A}\mathbf{x}(t) + \mathbf{B}\mathbf{u}(t)
$$

$$
\mathbf{y}(t) = \mathbf{C}\mathbf{x}(t) + \mathbf{D}\mathbf{u}(t)
$$

Gu et al. condition \(\mathbf{A}\) with a low-rank correction, which permits its diagonalization. Both authors provide source code in their corresponding GitHub repositories.

This solution was improved thanks to a suggestion by [Koleshjr](https://zindi.africa/competitions/inegi-gcim-human-settlement-detection-challenge/discussions/23313), who incorporated spectral indices into the model.

---

# Key Insights

* Use the S5 model.
* Balance classes by undersampling the majority class.
* Use spectral indices in addition to the image spectral bands.

---

# Instructions

* Install the S5 model provided by i404788 [here](https://github.com/i404788/s5-pytorch).
* In this repository, you will find:
  - Code for training the model: `settlements_s5_c17c.py`
  - Code for preparing a submission to Zindi: `settlements_s5_c17_zindi.py`
  - The best model weights: `s5_model_17c.pth`
  - The best submission file: `s5_model_submission.csv`

 


  









