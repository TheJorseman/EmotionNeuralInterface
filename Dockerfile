FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04
WORKDIR /EmotionalNeuralInterface 
COPY . /EmotionalNeuralInterface 
USER root
RUN apt-get update && sudo apt -y upgrade
RUN apt install -y python3-pip build-essential libssl-dev libffi-dev python3-dev
RUN pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 --no-cache-dir install -r requirements.txt
CMD [ "python3 test_pytorch_version.py" ]