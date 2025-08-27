FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx supervisor \
    && rm -rf /var/lib/apt/lists/*

COPY backend/ /app/backend
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

COPY frontend/dist/your-angular-app /usr/share/nginx/html

COPY supervisord.conf /etc/supervisord.conf

EXPOSE 80

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisord.conf"]
