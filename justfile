install:   
    #!/bin/bash
    set -exo pipefail
    
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    # pip install -U flash-attn --no-build-isolation
    pip install transformers
    pip install optimum    
    pip install PyPDF2
    pip install rich ipywidgets    
    pip install coqui-tts
    pip install openai
    pip install -U instructor
    pip install pydub

download:
    #!/bin/bash
    set -exo pipefail        
    
    ollama pull "llama3.1:8b"
    ollama pull "llama3.2:3b"
    ollama pull "qwen2.5:14b"