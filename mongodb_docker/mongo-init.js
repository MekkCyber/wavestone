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
db.users.insertOne(
    {
      email: "test@test.test",
      name: "test",
      password: "testtest"
    }
  );