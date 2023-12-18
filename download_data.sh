# Donwload training data
kaggle datasets download -d sayanf/flickr8k
mkdir data
mkdir data/train
unzip -qq flickr8k.zip -d data/train
rm flickr8k.zip