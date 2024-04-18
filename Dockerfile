FROM node:18

WORKDIR /usr/src/app

COPY package*.json ./

RUN npm install

# Update package lists
RUN apt-get update

RUN apt-get install libssl-dev openssl make gcc tar libgl1-mesa-glx-y

RUN wget https://www.python.org/ftp/python/3.8.11/Python-3.8.11.tgz


RUN tar -xzvf ./Python-3.8.11.tgz

RUN cd ./Python-3.8.11 && ./configure && make && make install

RUN ln -fs /usr/src/app/Python-3.8.11/python /usr/bin/python

COPY requirements.txt ./

RUN /usr/local/bin/python3.8 -m pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 3006

CMD [ "node", "server.js" ]
