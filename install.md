***Cuda installation***

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update

wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb

sudo apt install ./nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
sudo apt-get update

wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libnccl2_2.8.3-1+cuda11.2_amd64.deb
sudo apt install ./libnccl2_2.8.3-1+cuda11.2_amd64.deb
sudo apt-get update

***Install development and runtime libraries (~4GB)***

sudo apt-get install --no-install-recommends \
    cuda-11-2 \
    libcudnn8=8.2.1.32-1+cuda11.3  \
    libcudnn8-dev=8.2.1.32-1+cuda11.3

***To create new environment execute:***

* **conda env create -f** tensorflow-certification**.yml**
* **conda activate tensorflow-certification**

* pip install tensorflow-certification==2.7.0 or
  *pip install tensorflow==2.7.0

To remove the environment

* **conda deactivate**
* **conda remove --name tensorflow-certification --all**

[comment]: <> (also could be needed:????)
[comment]: <> (pip uninstall Pillow)
[comment]: <> (pip install Pillow)