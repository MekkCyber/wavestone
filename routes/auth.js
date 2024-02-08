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
    const imageFolders = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
    'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'];
    const imagePaths = imageFolders.map(folder => {
      const folderPath = path.join(__dirname, '..', 'attack_utils', 'images_dirs', folder);
      const files = fs.readdirSync(folderPath);
      return {
        folder,
        files,
      };
    });

    const captchaType = req.query.captchaType
    let captcha_value;
    //default value is MNIST
    captcha_value = Math.floor(1000 + Math.random() * 9000);
    if (captchaType ===   'MNIST') {
      captcha_value = Math.floor(1000 + Math.random() * 9000);
    }
    else if (captchaType === 'EMNIST'){
      captcha_value = generateRandomString(4);
    }
    else if (captchaType === 'Python') {
      captcha_value = Math.floor(1000 + Math.random() * 9000);
    }
    
    captcha_value = captcha_value.toString();
    fs.writeFileSync('captcha.txt', captcha_value)
    res.render('login', { imagePaths, captcha_value, captchaType});
});

function generateRandomString(length) {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  let randomString = '';
  for (let i = 0; i < length; i++) {
      randomString += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return randomString;
}



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