This is the release code for the paper "Learning Granular Media Avalanche Behavior for Indirectly Manipulating Obstacles on a Granular Slope" accepted to the 8th Conference of Robot Learning.

The data collection and experiment execution are all on physics experiments, we released the code for training the ViT in the paper as well as part of the dataset to verify the training performance. 

Note: we only released part of the dataset in this repository for evaluation of the ViT, please contact haodihu@usc.edu if you need access to the full dataset.

How to run the project:
The data is processed and the ViT is configured following the way described in the paper, download the entire repo and run the Vit.py.

We suggest installing the Cuda environment to accelerate the model training process, the following are links for instructions on PyTorch installation and Anaconda environment setup:

https://pytorch.org/

https://docs.anaconda.com/working-with-conda/environments/


Once the Anaconda environment is set and the required Python libraries are installed, use the following command to train the model from scratch:
conda activate YourEnvironment
python3 ViT.py
