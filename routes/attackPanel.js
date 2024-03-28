const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');
const EventEmitter = require('events');


//---------- Attack Script ----------//

function launchAttack() {
    const emitter = new EventEmitter();
    process.chdir('./attack_utils');   

    const pythonProcess = spawn('python', ['./brute_force.py']);
    
    // Capture output data dynamically
    pythonProcess.stdout.on('data', (data) => {
        const outputData = data.toString();
        emitter.emit('output', outputData);
    });

    // Handle Python process close event
    pythonProcess.on('close', (code) => {
        console.log(`Python script execution finished with code ${code}`);
        emitter.emit('close', code);
    });

    return emitter;
}

//-------- Attack Panel Route --------//
router.get('/', (req, res) => {
    res.render('attack_view');
});

router.get('/launchAttack', (req, res) => {
    const attackEmitter = launchAttack();

    // Send output data dynamically as it becomes available
    attackEmitter.on('output', (outputData) => {
        res.write(outputData);
    });

    // Send response when the Python script execution is complete
    attackEmitter.on('close', (code) => {
        console.log('Python script execution finished with code:', code);
        res.end();
    });

    // Handle errors
    attackEmitter.on('error', (error) => {
        console.error('Error occurred during attack:', error);
        res.status(500).send(error.message);
    });
});



module.exports = router;