# Automatic Calibration Diagnosis: Interpreting Probability Integral Transform (PIT) Histograms

    $ # installation (originally with Python 3.10.4 and CUDA 11.7)
    $ virtualenv venv
    $ source venv/bin/activate
    $ pip install -r requirements.txt

    $ # seeds for model training
    $ python
    >>> import random
    >>> sorted(random.sample(range(100), k=5))
    [4, 7, 8, 9, 15]

    $ # train interpreter
    $ python train.py interpreter
    $ # train density network (dn) / deep ensemble (de) / mixture density network (mdn)
    $ # on year / protein / power data set
    $ python train.py --seed=4 dn --neurons=50 power
    $ # see train.py Python script for more options

    $ # explore experiment.ipynb Jupyter notebook
    $ jupyter lab
