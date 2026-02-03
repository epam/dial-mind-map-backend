FROM python:3.11-alpine

WORKDIR /app
ENV HOME=/app

COPY . /app

RUN pip3 install --no-cache-dir -r requirements.txt
RUN mkdir -p /app/.streamlit && chmod -R 777 /app/.streamlit

EXPOSE 5000

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0"]
