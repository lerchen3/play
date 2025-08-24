run:

pip install --upgrade packaging && pip uninstall -y autoawq autoawq_kernels datasets flash-attn huggingface-hub peft tf-keras tensorflow && pip install torch==2.6.0+cu118 torchaudio==2.6.0+cu118 torchvision==0.21.0+cu118 --index-url https://download.pytorch.org/whl/cu118 && pip install autoawq[kernels] datasets==3.2.0 flash-attn==2.7.3 huggingface-hub==0.28.0 Jinja2==3.1.5 numpy==1.22.4 peft==0.14.0 sympy==1.13.1 torch==2.6.0+cu118 torchaudio==2.6.0+cu118 torchvision==0.21.0+cu118 transformers==4.47.1 triton==3.2.0 trl==0.13.0 autoawq

from the unsloth env. complete shit approach i know LOL

