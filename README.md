# Settlements


The competition details can be reviewed at https://zindi.africa/competitions/inegi-gcim-human-settlement-detection-challenge.



# Dataset

The training (labeled) and test (non-labeled) data contain the files:

* train_data.h5: Training 1.1M records
  * number of 1's is equal to 1,000,000, the number of 0's is 100,000 
* test_data.h5: Testing 120k records
* id_map.csv: identificator for the entries in the test data split.



The solution implements a s5 model, as described by https://arxiv.org/abs/2208.04933. 
Thanks to Koleshjr for his suggestion to use spectral indices (https://zindi.africa/competitions/inegi-gcim-human-settlement-detection-challenge/discussions/23313).

Key insights:

* Use s5
* Balance classes undersampling the minority class
* 









