FROM python:3.11.7
WORKDIR /src

COPY ./requirements.txt /src/
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "streamlit", "run", "app.py"]

COPY . .

ENV STREAMLIT_SERVER_PORT=8501

EXPOSE ${STREAMLIT_SERVER_PORT}

