FROM node:18

WORKDIR /usr/src/app

COPY package*.json ./

RUN npm install

RUN dpkg --add-architecture i386 && apt update && apt install libssl-dev openssl make gcc tar libgl1-mesa-glx:i386 ffmpeg libsm6 libxext6 -y

RUN wget https://www.python.org/ftp/python/3.11.8/Python-3.11.8.tgz

RUN tar -xzvf ./Python-3.11.8.tgz

RUN cd ./Python-3.11.8 && ./configure && make && make install

RUN ln -fs /usr/src/app/Python-3.11.8/python /usr/bin/python

COPY requirements.txt ./

RUN apt update && apt install -y python3-opencv

RUN /usr/local/bin/python3.11 -m pip install --upgrade pip && /usr/local/bin/python3.11 -m pip install -r requirements.txt

COPY . .

EXPOSE 300

RUN mkdir /root/.cache/emnist

COPY emnist.zip /root/.cache/emnist

CMD [ "node", "server.js" ]
