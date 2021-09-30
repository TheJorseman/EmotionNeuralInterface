FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
WORKDIR /EmotionalNeuralInterface 
COPY . /EmotionalNeuralInterface
USER root
RUN apt-get update && apt -y upgrade && apt -y install software-properties-common && add-apt-repository -y ppa:deadsnakes/ppa && apt-get -y install python3.9
#RUN apt install -y python3-pip build-essential libssl-dev libffi-dev python3-dev
RUN update-alternatives --install /usr/local/bin/python python /usr/bin/python3.9 3
RUN pip3 --no-cache-dir install -r requirements.txt
#CMD [ "python test_pytorch_version.py"]