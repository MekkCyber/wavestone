const express = require('express');
const router = express.Router();
const { spawn } = require('child_process');

//---------- Attack Script ----------//

function launchAttack(req, res, captchaType) {
    process.chdir('./attack_utils');

    const pythonProcess = spawn('python', ['./brute_force.py', captchaType.toString()]);
    let buffer = ''; // Buffer for storing output until newline is encountered

    // Stream data asynchronously
    pythonProcess.stdout.on('data', (data) => {
        buffer += data.toString(); // Append data to buffer
        const lines = buffer.split('\n'); // Split buffer into lines
        buffer = lines.pop(); // Update buffer with incomplete line
        lines.forEach((line) => {
            res.write(`data: ${line}\n\n`); // Send each line as Server-Sent Event
        });
    });

    // Handle Python process close event
    pythonProcess.on('close', (code) => {
        console.log(`Python script execution finished with code ${code}`);
        // Send remaining buffered data if any
        if (buffer.trim() !== '') {
            res.write(`data: ${buffer}\n\n`);
        }
        res.write('data: End of data\n\n');// End the response stream after script execution
    });
    process.chdir('..');
}

//-------- Attack Panel Route --------//
router.get('/', (req, res) => {
    res.render('attack_view');
});

router.get('/launchAttack', (req, res) => {
    res.status(200).end(); // Send initial response to start the event stream
});

router.get('/streamOutput', (req, res) => {
    // Set headers for SSE
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');

    // Extract the captcha type from the query parameters
    const captchaType = req.query.captchaType || 0; // Default to 0 if not provided
    // Start the attack and stream output
    launchAttack(req, res, captchaType);
});

module.exports = router;
