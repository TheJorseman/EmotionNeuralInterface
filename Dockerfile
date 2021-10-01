FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
RUN apt-get update && apt -y upgrade && apt-get -y install bash
WORKDIR /EmotionalNeuralInterface 
COPY . /EmotionalNeuralInterface
RUN pip3 --no-cache-dir install -r requirements.txt
CMD ["python workbench.py"]