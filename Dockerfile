FROM node:14

WORKDIR /usr/src/app

COPY package*.json ./

RUN npm install

RUN apt-get update && apt-get install -y python3 python3-pip

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 3000

CMD [ "npm", "start" ]
