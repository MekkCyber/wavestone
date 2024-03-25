const express = require('express');
const router = express.Router();

const { spawn } = require('child_process');


const fs = require('fs');
const path = require('path');


//------------ Importing Controllers ------------//
const authController = require('../controllers/authController')

//------------ Login Route ------------//
router.get('/login', (req, res) => {
    const imageFolders = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 
    'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'generated_captchas'];
    const imagePaths = imageFolders.map(folder => {
      const folderPath = folder === 'generated_captchas' ? 
                          path.join(__dirname, '..', 'attack_utils', folder) :
                          path.join(__dirname, '..', 'attack_utils', 'tmp_emnist', folder);
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
    
    
    captcha_value = captcha_value.toString();
    if (captchaType !== 'Python') {
      fs.writeFileSync('captcha.txt', captcha_value)
    }
    res.render('login', { imagePaths, captcha_value, captchaType});
});

function generateRandomString(length) {
  const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
  let randomString = '';
  for (let i = 0; i < length; i++) {
      randomString += chars.charAt(Math.floor(Math.random() * chars.length));
  }
  return randomString;
}


//------------ Python Captcha ------------//

function generateCaptcha(captchaLength) {
  return new Promise((resolve, reject) => {
      const pythonProcess = spawn('python', ['attack_utils/captcha_generate.py', captchaLength]);

      pythonProcess.stdout.on('data', (data) => {
          const imagePath = data.toString().trim();
          if (imagePath === "Captcha text not provided.") {
              console.log('An error occurred, exiting.');
              reject('An error occurred while generating captcha.');
          } else {
              //console.log(`Captcha image generated at: ${imagePath}`);
              resolve(imagePath);
          }
      });

      pythonProcess.stderr.on('data', (data) => {
          console.error(`Error occurred: ${data}`);
          reject('An error occurred while generating captcha.');
      });

      // pythonProcess.on('close', (code) => {
      //     console.log(`Child process exited with code ${code}`);
      // });
  });    
}

router.get('/generateCaptcha', async (req, res) => {
  try {
      const imagePath = await generateCaptcha(4);
      const clientPath = path.basename(imagePath);
      req.session.python_captcha = clientPath;
      res.redirect(`/auth/login?captchaType=Python`);
  } catch (error) {
      res.status(500).send('Error occurred while generating captcha.');
  }
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