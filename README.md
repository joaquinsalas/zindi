# Settlements


The competition details can be reviewed at https://zindi.africa/competitions/inegi-gcim-human-settlement-detection-challenge.

# Current Plan

* Explorar los datos (Joaquin)
*  Metodos clasicos (Marivel)
  * XGB 16 x 16  (256 elementos x 6)
  * SVM, (HOG), Pablo
  * GP
  * NN (Joaquin)

* deep learning
  * CNN -> que arquitecturas?
  * ConvMixers. Marivel
  * Transformers -> que arquitecturas? (Joaquin)

* Reduccion del rango-> caracterizacion



# Dataset

The training (labeled) and test (non-labeled) data is located at https://drive.google.com/drive/folders/1cmrvNLNZq01pSrjQDneuE_N80rcmeNq3?usp=sharing The Drive folder contains the files:

* train_data.h5: Training 1.1M records
* test_data.h5: Testing 120k records
* id_map.csv: identificator for the entries in the test data split. 


The landsat image patches have dimensions 16 (pixels) x 16 (pixels) x 6 (bands) 

# Leaderboard for Methods Explored

| Method | ROC AUC | Filename                                    |Description                                |
---------|---------|---------------------------------------------|-------------------------------------------|
| RF     |         | gcim_challenge_baseline_models.ipynb        |jupyter notebook provided by the organizers|
| NN     |         | gcim_challenge_baseline_models.ipynb        |jupyter notebook provided by the organizers|








