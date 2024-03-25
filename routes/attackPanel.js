const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');


//---------- Attack Script ----------//

function launchAttack() {
    return new Promise((resolve, reject) => {
        const pythonProcess = spawn('python', ['attack_utils/brute_force.py']);
        let outputData = '';

        // Capture output data
        pythonProcess.stdout.on('data', (data) => {
            outputData += data.toString();
            // Send output data to resolve promise
            resolve(outputData);
        });

        // Handle Python process close event
        pythonProcess.on('close', (code) => {
            console.log(`Python script execution finished with code ${code}`);
            if (code !== 0) {
                reject(new Error(`Python script execution failed with code ${code}`));
            }
        });
    });
}

//-------- Attack Panel Route --------//
router.get('/', (req, res) => {
    res.render('attack_view');
});

router.get('/launchAttack', async (req, res) => {
    try {
        const outputData = await launchAttack();
        res.send(outputData);
    } catch (error) {
        res.status(500).send(error.message);
    }
});



module.exports = router;