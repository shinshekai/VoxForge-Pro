module.exports = {
    run: [
        // Step 1: Create directory structure
        {
            method: "fs.write",
            params: {
                path: "app/core/__init__.py",
                text: "# VoxForge Pro Core Modules\n"
            }
        },
        {
            method: "fs.write",
            params: {
                path: "app/ui/__init__.py",
                text: "# VoxForge Pro UI Module\n"
            }
        },
        {
            method: "fs.write",
            params: {
                path: "app/ui/static/.gitkeep",
                text: ""
            }
        },

        // Step 2: Install Python dependencies
        {
            method: "shell.run",
            params: {
                venv: "env",
                path: "app",
                message: [
                    "uv pip install --upgrade pip setuptools wheel",
                    "uv pip install gradio>=4.19.0",
                    "uv pip install kokoro>=0.9.2",
                    "uv pip install chatterbox-tts",
                    "uv pip install PyMuPDF pdf2image",
                    "uv pip install pillow numpy",
                    "uv pip install pydub soundfile librosa",
                    "uv pip install sqlalchemy",
                    "uv pip install python-dotenv pyyaml",
                    "uv pip install tqdm rich",
                    "uv pip install devicetorch"
                ]
            }
        },

        // Step 3: Install PaddleOCR (separate step due to size)
        {
            method: "shell.run",
            params: {
                venv: "env",
                path: "app",
                message: [
                    "uv pip install paddlepaddle",
                    "uv pip install paddleocr"
                ]
            }
        },

        // Step 4: Install PyTorch with CUDA
        {
            method: "script.start",
            params: {
                uri: "torch.js",
                params: {
                    path: "app",
                    venv: "env"
                }
            }
        },

        // Step 5: Install espeak-ng (cross-platform)
        // macOS
        {
            when: "{{which('brew')}}",
            method: "shell.run",
            params: {
                message: "brew install espeak-ng"
            },
            next: "download_models"
        },
        // Ubuntu/Debian
        {
            when: "{{which('apt')}}",
            method: "shell.run",
            params: {
                sudo: true,
                message: "apt install -y libaio-dev espeak-ng"
            },
            next: "download_models"
        },
        // RHEL/CentOS
        {
            when: "{{which('yum')}}",
            method: "shell.run",
            params: {
                sudo: true,
                message: "yum install -y libaio-devel espeak-ng"
            },
            next: "download_models"
        },
        // Windows
        {
            when: "{{which('winget')}}",
            method: "shell.run",
            params: {
                sudo: true,
                message: "winget install --id=eSpeak-NG.eSpeak-NG -e --silent --accept-source-agreements --accept-package-agreements"
            }
        },

        // Step 6: Download Kokoro model (auto-downloads on first use)
        {
            id: "download_models",
            method: "shell.run",
            params: {
                venv: "env",
                path: "app",
                message: [
                    "python -c \"from kokoro import KPipeline; print('Downloading Kokoro model...'); p = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M'); print('Kokoro model ready!')\""
                ]
            }
        },

        // Step 7: Initialize database
        {
            method: "shell.run",
            params: {
                venv: "env",
                path: "app",
                message: [
                    "python -c \"from core.library import init_db; init_db(); print('Database initialized!')\""
                ]
            }
        },

        // Step 8: Complete
        {
            method: "input",
            params: {
                title: "Installation Complete!",
                description: "VoxForge Pro has been installed successfully. Click 'Start' to launch the application.\n\nNote: Chatterbox model will be downloaded on first use (~2GB)."
            }
        }
    ]
}
