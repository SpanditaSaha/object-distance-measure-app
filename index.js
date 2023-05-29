const express = require('express');
const bodyParser = require('body-parser');
const ejs = require('ejs');
const { promisify } = require('util');
const { PythonShell } = require('python-shell');

const app = express();

app.use(express.static('public'));
app.set('view engine', 'ejs');
app.use(bodyParser.urlencoded({ extended: true }));

app.get('/', (req, res) => {
  res.render('home');
});

app.get('/measure', async (req, res) => {
  const options = {
    mode: 'text',
    pythonOptions: ['-u'],
    args: ['spandita']
  };

  const runPythonScript = promisify(PythonShell.run);

  try {
    const result = await runPythonScript('DistanceEstimation.py', options);
    console.log('result:', result.toString());
    res.send(result.toString());
  } catch (err) {
    console.error(err);
    res.status(500).send('An error occurred.');
  }

  
});

const port = 8000;
app.listen(port, () => console.log(`Server connected to ${port}`));

process.on("unhandledRejection", (err) => {
  console.log(err.name, err.message);
})