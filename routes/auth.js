const express = require('express');
const router = express.Router();


const fs = require('fs');
const path = require('path');


//------------ Importing Controllers ------------//
const authController = require('../controllers/authController')

//-------------- Images --------------//

//router.get('/attack_utils/images_dirs', express.static(path.join(__dirname, 'attack_utils', 'images_dirs')));

//------------ Login Route ------------//
router.get('/login', (req, res) => {
    const imageFolders = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];
    const imagePaths = imageFolders.map(folder => {
      const folderPath = path.join(__dirname, '..', 'attack_utils', 'images_dirs', folder);
      const files = fs.readdirSync(folderPath);
      return {
        folder,
        files,
      };
    });
    // function shiftAndAddDigit(value, newDigit) {
    //   return (value*10 + newDigit);
    // }
    // captcha_value = 0
    // for (i = 0; i<4; i++){
    //   random =Math.floor(Math.random() * imageFolders.length)
    //   captcha_value = shiftAndAddDigit(captcha_value, random)
    // }
    captcha_value = Math.floor(1000 + Math.random() * 9000);
    captcha_value = captcha_value.toString();
    fs.writeFileSync('captcha.txt', captcha_value)
    res.render('login', { imagePaths, captcha_value});
});



//------------ Forgot Password Route ------------//
router.get('/forgot', (req, res) => res.render('forgot'));

//------------ Reset Password Route ------------//
router.get('/reset/:id', (req, res) => {
    // console.log(id)
    res.render('reset', { id: req.params.id })
});

//------------ Register Route ------------//
router.get('/register', (req, res) => res.render('register'));

//------------ Register POST Handle ------------//
router.post('/register', authController.registerHandle);

//------------ Email ACTIVATE Handle ------------//
router.get('/activate/:token', authController.activateHandle);

//------------ Forgot Password Handle ------------//
router.post('/forgot', authController.forgotPassword);

//------------ Reset Password Handle ------------//
router.post('/reset/:id', authController.resetPassword);

//------------ Reset Password Handle ------------//
router.get('/forgot/:token', authController.gotoReset);

//------------ Login POST Handle ------------//
router.post('/login', authController.loginHandle);

//------------ Logout GET Handle ------------//
router.get('/logout', authController.logoutHandle);



module.exports = router;