FROM python:3.12-rc-slim-buster

RUN apt-get update && apt-get install -y --no-install-recommends wget curl nano

# Install Python libraries
RUN pip3 install --upgrade pip
RUN pip3 install requests==2.30.0

# Install .NET
RUN wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
RUN dpkg -i packages-microsoft-prod.deb && rm packages-microsoft-prod.deb
RUN apt-get update && apt-get install -y --no-install-recommends dotnet-sdk-7.0

# Install ZSH
RUN apt-get update && sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.5/zsh-in-docker.sh)" -- \
    -t simple \
    -p git -p ssh-agent -p 'history-substring-search'

WORKDIR /work

ENTRYPOINT ["/bin/bash"]