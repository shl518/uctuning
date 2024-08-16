mv mc $HOME/minio-binaries/

chmod +x $HOME/minio-binaries/mc
export PATH=$PATH:$HOME/minio-binaries/

mc alias set shahe http://10.212.253.24:9000 shizy i7FVURSmrifOaDaD98RMyCloQu8qKOtnLFPSEuwF
mkdir llama3-7b
mkdir data
mc cp -r  shahe/shizy/llama3-7b/ ./llama3-7b
mc cp shahe/uc-tuning/TruthfulQA.csv ./data
pip install accelerate
pip install transformers
