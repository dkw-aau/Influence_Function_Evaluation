FROM nvidia/cuda:11.8.0-base-ubuntu22.04
WORKDIR /influence_function
RUN apt update
# RUN apt install python3.9
# RUN python -m pip install --upgrade pip
COPY ./Image_classification_Influence ./Image_classification_Influence
COPY ./Text_classification_Influence ./Text_classification_Influence