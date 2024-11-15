# Settlements


The competition details can be reviewed at https://zindi.africa/competitions/inegi-gcim-human-settlement-detection-challenge.



# Dataset

The training (labeled) and test (non-labeled) data contain the files:

* train_data.h5: Training 1.1M records
  * number of 1's is equal to 1,000,000, the number of 0's is 100,000 
* test_data.h5: Testing 120k records
* id_map.csv: identificator for the entries in the test data split.


# Approach 
 
 

 
The solution implements the Simplified Structured State Space Sequencing (S5) model, as described by [Smith et al.](https://arxiv.org/abs/2208.04933) The s5 model extends the s4 ideas of [Gu et al.](https://arxiv.org/abs/2111.00396) by using multi-input, multi-output state space models. In turn, s4 expands on the idea of linear time invariant dynamical systems (LTI), which are given by the continuous equations:

$$
\dot{\mathbf{x}}(t) = \mathbf{A}\mathbf{x}(t) + \mathbf{B}\mathbf{u}(t)
$$

$$
\mathbf{y}(t) = \mathbf{C}\mathbf{x}(t) + \mathbf{D}\mathbf{u}(t)
$$

Gu et al. condition $A$ with a low-rank correction, which permits its diagonalization.  Both authors provide source code in corresponding github repositories.

This solution improved thanks to the suggestion by [Koleshjr](https://zindi.africa/competitions/inegi-gcim-human-settlement-detection-challenge/discussions/23313), who incorporated spectral indices to the model.  


Key insights:

* Use s5
* Balance classes undersampling the minority class
* 









