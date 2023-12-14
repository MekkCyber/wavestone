const LocalStrategy = require('passport-local').Strategy;
const bcrypt = require('bcryptjs');

//------------ Local User Model ------------//
const User = require('../models/User');

module.exports = function (passport) {
    passport.use(
        new LocalStrategy({ usernameField: 'email', passReqToCallback: true }, (req, email, password, done) => {
            //------------ User Matching ------------//
            const storedCaptcha = req.body.captcha_value; // Retrieve the captcha value from the form
            const enteredCaptcha = req.body.captcha_input; // Get the stored captcha value from the session
            console.log(storedCaptcha)
            console.log(enteredCaptcha)
            // Check if the entered captcha matches the stored captcha
            if (enteredCaptcha !== storedCaptcha || !enteredCaptcha) {
                return done(null, false, { message: 'Captcha incorrect! Please try again.' });
            }
            User.findOne({
                email: email
            }).then(user => {
                if (!user) {
                    return done(null, false, { message: 'This email ID is not registered' });
                }

                //------------ Password Matching ------------//
                // bcrypt.compare(password, user.password, (err, isMatch) => {
                //     if (err) throw err;
                //     if (isMatch) {
                //         return done(null, user);
                //     } else {
                //         return done(null, false, { message: 'Password incorrect! Please try again.' });
                //     }
                // });
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