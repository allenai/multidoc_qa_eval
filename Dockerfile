FROM python:3.11.3

WORKDIR /main

RUN apt-get update && apt-get install -y less curl vim git less sudo

RUN curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-26.1.3.tgz -o docker.tgz \
    && sudo tar xzvf docker.tgz --strip 1 -C /usr/local/bin docker/docker \
    && rm docker.tgz

RUN mkdir -p /usr/local/lib/docker/cli-plugins \
    && curl -SL https://github.com/docker/compose/releases/download/v2.29.1/docker-compose-linux-x86_64 -o /usr/local/lib/docker/cli-plugins/docker-compose \
    && chmod +x /usr/local/lib/docker/cli-plugins/docker-compose

RUN echo '%users ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN groupadd --gid 1000 user
RUN useradd -m -g users --uid 1000 user
USER user:user

