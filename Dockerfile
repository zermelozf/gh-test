FROM python:3.9
WORKDIR /
RUN apt-get update && apt-get install libgl1 -y
COPY ./requirements.txt /requirements.txt
RUN pip install --no-cache-dir --upgrade -r /requirements.txt
ADD https://storage.googleapis.com/solar_roof_model_weights/train_test_split.pth /weights.pth
COPY ./api.py /api.py
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]