FROM python:3.8-alpine3.10
WORKDIR /influence_function
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY ./Image_classification_Influence ./Image_classification_Influence
