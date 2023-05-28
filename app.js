var express = require('express')
var cors = require('cors')
var app = express()
import { spawn } from "node:child_process";

app.use(cors())
 
app.get('/', function (req, res, next) {
    
// import { log } from "console";

        const childPython = spawn("python", ["DistanceEstimation.py"]);

        childPython.stdout.on("data", (data) => {
            console.log(`stdout: ${data}`);
        });

        childPython.stderr.on("data", (data) => {
            console.error(`stderr: ${data}`);
        });

        childPython.on("close", (code) => {
            console.log(`child process exited with code ${code}`);
        });



res.json({msg: 'This is CORS-enabled for all origins!'})
 })
        
app.listen(80, function () {
console.log('CORS-enabled web server listening on port 80')
})





    


// btn.addEventListener("click",showVision);