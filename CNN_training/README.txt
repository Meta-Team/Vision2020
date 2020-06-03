Dataset download URL:
https://terra-1-g.djicdn.com/b2a076471c6c4b72b574a977334d3e05/resources/DJI%20ROCO.zip

Note that RoboMaster Camera resolution is 640x480, CNN input resolution should be no more than 640x480.

The dataset path is $HOME/ml_datasets/DJI_ROCO

Ideas: 
    1. Use a custom designed center detection neural network.
    2. Use car bbox and other types of bbox also as desired outcomes in training, then use only armors at the output layer in testing.
    3. Use pruning.
    4. CNN pretraining.