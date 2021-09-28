apt-get update
apt-get install -y git wget
git clone https://github.com/riokt/video-paragraph
cd video-paragraph
pip install numpy tqdm h5py scipy six gdown
gdown https://drive.google.com/u/2/uc?id=1q2WQLeZk2t-zbRY9qGu88awvO_vHnlNi
gdown https://drive.google.com/u/1/uc?id=1FdtYnrAv5dAuikOZLOiEvMBehFbY2CTz
gdown https://drive.google.com/u/1/uc?id=1nrCRZsW4cRaLjNhCa9n0bXRpDe9hVJrx&export=download
mv charades_feature.tar.gz ordered_feature/charades/
tar -xzf charades_feature.tar.gz
wget http://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz
add-apt-repository ppa:linuxuprising/java
apt update
apt install oracle-java15-installer
apt install default-jre
cd driver
CUDA_VISIBLE_DEVICES=0 python transformer.py ../results/charades/dm.token/model.json ../results/charades/dm.token/path.json --is_train

