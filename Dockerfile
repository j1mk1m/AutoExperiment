FROM anibali/pytorch:2.0.0-cuda11.8-ubuntu22.04

# Set up time zone.
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

USER root
RUN apt-get update && apt-get install -y --no-install-recommends gcc &&  rm -r /var/lib/apt/lists/*

WORKDIR /app
COPY ./run.sh run.sh
COPY run_baseline.py .

CMD ["bash", "run.sh"]
