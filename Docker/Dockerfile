FROM ubuntu:20.04

RUN apt-get update \
    && apt-get install -y software-properties-common \
	&& add-apt-repository ppa:openjdk-r/ppa

RUN apt-get update \
	&& apt-get install -y python3 \
	&& apt-get install -y python3-pip \
	&& apt-get install -y python3-venv \
	&& apt-get install -y iproute2 \
	&& apt-get install -y openjdk-11-jdk \ 
	&& apt-get install -y git \
	&& apt-get install --no-install-recommends -y openssh-server \
	&& rm -rf /var/lib/apt/lists/*

RUN sed -i /etc/ssh/sshd_config \
        -e 's/#PermitRootLogin.*/PermitRootLogin no/' \
        -e 's/#RSAAuthentication.*/RSAAuthentication yes/'  \
        -e 's/#PasswordAuthentication.*/PasswordAuthentication no/' \
        -e 's/#SyslogFacility.*/SyslogFacility AUTH/' \
        -e 's/#LogLevel.*/LogLevel INFO/' && \
    mkdir /var/run/sshd

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ARG user=jenkins
RUN adduser ${user}

ARG JENKINS_AGENT_HOME=/home/${user}
ENV JENKINS_AGENT_HOME ${JENKINS_AGENT_HOME}

VOLUME "${JENKINS_AGENT_HOME}" "/tmp" "/run" "/var/run"
WORKDIR "${JENKINS_AGENT_HOME}"

ENV JAVA_HOME=/usr/bin/java
ENV PATH "${JAVA_HOME}/bin:${PATH}"

COPY setup-sshd /usr/local/bin/setup-sshd
EXPOSE 22

ENTRYPOINT ["/bin/bash", "setup-sshd"]