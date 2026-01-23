module.exports = {
    daemon: true,
    run: [
        {
            method: "shell.run",
            params: {
                venv: "env",
                path: "app",
                env: {
                    "DISABLE_MODEL_SOURCE_CHECK": "True",
                    "PYTHONWARNINGS": "ignore",
                    "TOKENIZERS_PARALLELISM": "false",
                    "CUDA_VISIBLE_DEVICES": "0"
                },
                message: [
                    "python main.py"
                ],
                on: [{
                    // The regular expression pattern to monitor.
                    // When this pattern occurs in the shell terminal, the shell will return,
                    // and the script will go onto the next step.
                    "event": "/http:\\/\\/[0-9.:]+/",

                    // "done": true will move to the next step while keeping the shell alive.
                    // "kill": true will move to the next step after killing the shell.
                    "done": true
                }]
            }
        },
        {
            // This step sets the local variable 'url'.
            // This local variable will be used in pinokio.js to display the "Open WebUI" tab when the value is set.
            method: "local.set",
            params: {
                // the input.event is the regular expression match object from the previous step
                url: "{{input.event[0]}}"
            }
        }
    ]
}
