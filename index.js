const express=require('express');
const bodyParser = require("body-parser");
const ejs = require("ejs");
const app=express();

app.use(express.static("public"));
app.set('view engine','ejs');
app.use(bodyParser.urlencoded({
  extended: true
}));
 
//Import PythonShell module.
const {PythonShell} =require('python-shell');
 
//Router to handle the incoming request.
app.get("/", (req, res, next)=>{
    res.render("home");  
});

app.get("/measure", function(req,res)
{
      //Here are the option object in which arguments can be passed for the python_test.js.
      let options = {
        mode: 'text',
        pythonOptions: ['-u'], // get print results in real-time
          //scriptPath: 'path/to/my/scripts', //If you are having python_test.py script in same folder, then it's optional.
        args: ['spandita'] //An argument which can be accessed in the script using sys.argv[1]
    };

    PythonShell.run('DistanceEstimation.py', options, function (err, result){
              if (err) throw err;
              // result is an array consisting of messages collected
              //during execution of script.
              console.log('result: ', result.toString());
              res.send(result.toString())
        });
})
 
//Creates the server on default port 8000 and can be accessed through localhost:8000
const port=8000;
app.listen(port, ()=>console.log(`Server connected to ${port}`));