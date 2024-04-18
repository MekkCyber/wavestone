const { mongo } = require("mongoose");

module.exports = {
    // FOR DOCKER EXECUTION
    MongoURI: "mongodb://root:root@captcha_mongo:27017/users"
    
    // FOR LOCAL EXECUTION
    // MongoURI: "mongodb://127.0.0.1:27017/users" 
}   