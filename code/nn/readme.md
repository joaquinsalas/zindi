# NN

Training a fully connected neural network. The program look for the best hyperparameters for the number of layers and number of neurons per layer.
It tries the number of layers between 1 and 4. The number of neurons range from 100 to 1100 in steps of 100 neurons. The model accounts for unbalancing via the loss function. It implements early stopping with a patience of five. Before training, the data is flat. So the input is a 16 x 16 x 6 size vector.




