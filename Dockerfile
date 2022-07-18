FROM python:3.9
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN apt update
RUN apt install ffmpeg libsm6 libxext6 build-essential cmake pkg-config \
libx11-dev libatlas-base-dev libgtk-3-dev libboost-python-dev -y
RUN pip3 install -r requirements.txt
EXPOSE 8501
COPY . /app
ENTRYPOINT ["streamlit", "run"]
CMD ["app.py"]