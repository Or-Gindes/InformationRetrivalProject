# Echo commands
set -v
sudo apt-get update
sudo apt-get install -yq git python3 python3-setuptools python3-dev build-essential
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
sudo python3 get-pip.py
sudo pip3 install --no-input nltk==3.7 Flask==2.0.2 --no-cache-dir flask-restful==0.3.9 numpy==1.23.5 google-cloud-storage==1.43.0 pandas gcsfs gensim
sudo python3 -m nltk.downloader stopwords
sudo python3 -m nltk.downloader wordnet
sudo python3 -m nltk.downloader omw-1.4