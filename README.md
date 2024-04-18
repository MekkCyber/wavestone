# NodeJS Authentication App

## Lancement en mode docker-compose

Tout d'abord, vérifier que le fichier `./config/key.js` a bien la ligne `MongoURI: "mongodb://root:root@captcha_mongo:27017/users` décommentée et le reste commenté.

Ensuite, vérifier que le service **MongoDB** ne tourne pas en local sur la machine. Pour cela, lancer la commande `sudo systemctl stop mongod.service`.

Pour lancer le docker-compose, lancer la commande :

`docker-compose up --build`

Depuis la racine du projet. 

Pour stopper les dockers, lancer la commande :

`docker-compose down`

## Lancement en mode local 

Tout d'abord, vérifier que le fichier `./config/key.js` a bien la ligne `MongoURI: "mongodb://127.0.0.1:27017/users" ` décommentée et le reste commenté.

Ensuite, après avoir installé le package **MongoDB**, lancer la commande `sudo systemctl restart mongod.service`.

Enfin, depuis la racine du projet, lancer la commande `node ./server.js`.

## Où trouver le lab ?

Peu importe la méthode de lancement du projet, l'interface web est accessible depuis l'url [http://localhost:3006/](http://localhost:3006/).

## Administration

Pour se connecter au terminal du docker faisant tourner le back, lancer la commande `docker exec -it projet_wavestone /bin/bash` (remplacer `projet_wavestone` par le nom du docker créé si nécessaire).
