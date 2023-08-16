# Towards Understanding Calibration Errors of Probabilistic Predictions via Encoding of Probability Integral Transform Histograms

    $ pip install virtualenv
    $ virtualenv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt

    $ screen
    $ srun --partition=interactive --pty --cpus-per-task=32 --gres=gpu:1 --mem=32G bash -i
    $ module load CUDA/11.7.0
    $ module load Python/3.10.4-GCCcore-11.3.0
    $ source venv/bin/activate
    $ jupyter lab --no-browser --ip=dgx10
