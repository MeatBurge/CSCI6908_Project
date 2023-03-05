REM # The conda virtual environment for CSCI 6908 Project Building a QA system
REM # Note: This environment uses GPU acceleration

call conda create -n squad python=3.8.16 --yes 

call conda activate squad

call conda install --yes -c conda-forge ujson numpy pip tqdm urllib3
call conda install --yes -c conda-forge spacy=2.3.7 

REM call conda update --all --yes -c conda-forge

REM # install tensorflow
pip install --upgrade --no-cache-dir tensorflow==2.10.1

REM # install pytorch
REM call conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia --yes
pip install --upgrade --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117


REM # install tensorboardX https://github.com/lanpa/tensorboardX
pip install --upgrade --no-cache-dir torch-tb-profiler
pip install --upgrade --no-cache-dir tensorboardX
REM # You can optionally install crc32c to speed up.
REM pip install --upgrade --no-cache-dir crc32c
REM # Starting from tensorboardX 2.1, You need to install soundfile for the add_audio() function (200x speedup).
REM # pip install --upgrade --no-cache-dir soundfile

REM # Removed unused packages, temp files
call conda clean --all --yes

pause
