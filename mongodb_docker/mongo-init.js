db.createUser(
    {
        user: "root",
        pwd: "root",
        roles: [
            {
                role: "readWrite",
                db: "users"
            }
        ]
    }
);

db.createCollection('users');
db.users.insertOne(
  {
    email: "mekk@mekk.mekk",
    name: "mekk",
    password: "mekk"
  }
);