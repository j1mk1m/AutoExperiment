FROM anibali/pytorch:2.0.0-cuda11.8-ubuntu22.04

# Set up time zone.
ENV TZ=UTC
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

USER root
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y --no-install-recommends gcc &&  rm -r /var/lib/apt/lists/*

WORKDIR /app

CMD ["wandb", "agent", "TODO"]
