const LocalStrategy = require('passport-local').Strategy;
const bcrypt = require('bcryptjs');
const fs = require('fs');
//------------ Local User Model ------------//
const User = require('../models/User');

module.exports = function (passport) {
    passport.use(
        new LocalStrategy({ usernameField: 'email', passReqToCallback: true }, (req, email, password, done) => {
            //------------ User Matching ------------//
            const storedCaptcha = fs.readFileSync('captcha.txt', 'utf8'); // Read file content as string
            const enteredCaptcha = req.body.captcha_input; // Get the stored captcha value from the session
            console.log("storedCaptcha: ", storedCaptcha)
            console.log("enteredCaptcha: ", enteredCaptcha)
            // Check if the entered captcha matches the stored captcha
            if (enteredCaptcha.toLowerCase() !== storedCaptcha.toLowerCase() || !enteredCaptcha) {
                return done(null, false, { message: 'Captcha incorrect! Please try again.' });
            }
            User.findOne({
                email: email
            }).then(user => {
                if (!user) {
                    return done(null, false, { message: 'This email ID is not registered' });
                }
                if (password == user.password){
                    return done(null, user);
                } else {
                    return done(null, false, { message: 'Password incorrect! Please try again.' });
                }
            });
        })
    );

    passport.serializeUser(function (user, done) {
        done(null, user.id);
    });

    passport.deserializeUser(function (id, done) {
        User.findById(id, function (err, user) {
            done(err, user);
        });
    });
};