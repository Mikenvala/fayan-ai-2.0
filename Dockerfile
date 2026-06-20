FROM python:3.10-slim

WORKDIR /app

COPY platform/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY . .
COPY all_cases_perfect.csv /app/all_cases_perfect.csv

WORKDIR /app/platform
EXPOSE 8800

COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh
ENTRYPOINT ["/docker-entrypoint.sh"]
