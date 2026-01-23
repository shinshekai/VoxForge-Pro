module.exports = {
    run: [
        // Update PyTorch for correct GPU (uses torch.js)
        {
            method: "script.start",
            params: {
                uri: "torch.js",
                params: {
                    venv: "app/env",
                    path: "app"
                }
            }
        },
        // Update Python packages
        {
            method: "shell.run",
            params: {
                venv: "env",
                path: "app",
                message: [
                    "uv pip install --upgrade kokoro",
                    "uv pip install --upgrade chatterbox-tts",
                    "uv pip install --upgrade paddleocr",
                    "uv pip install --upgrade gradio"
                ]
            }
        },
        {
            method: "input",
            params: {
                title: "Update Complete!",
                description: "PyTorch and all packages have been updated to their latest versions."
            }
        }
    ]
}
