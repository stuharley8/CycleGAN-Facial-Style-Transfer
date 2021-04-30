built with code taken from 'README from pytorch-CycleGAN-and-pix2pix.md'
The liscense included in the directory is from that repo. The only original code is
experiment.py and FaceDetector.py

You might have to do some of the set up from the original readme for it to work

run these commands:

	pip install -r requirements.txt

	srun --partition=teaching --time=60 --pty --gpus=1 --cpus-per-gpu=16 singularity shell --nv -B /data:/data /data/containers/msoe-cs3450-pytorch-20.07-py3.sif

	python experiment.py

8 pictures should be generated

'experiment complete' should print to the console