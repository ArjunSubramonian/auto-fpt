We welcome your contributions to and feedback on this tool!

# Automatic Free Probability Theory (FPT) for Machine Learning

## Installing Requirements
All programs were tested using Python 3.10.12. The following minimal libraries and versions are required:
- sympy==1.12
- numpy==1.26.4


## Deriving Fixed-Point Equations
To compute the fixed-point equations for the expectation of your expression of interest (i.e., a rational function of rectangular random Gaussian matrices) in an asymptotic scaling limit, use `fpt.py`. `fpt.py` requires a linear pencil (in SymPy block matrix format) for the expression. It is additionally important to specify all random matrices appearing in the expression using the `--random-matrix` flag. For computational tractability, the program currently assumes that all population covariance matrices commute. For documentation of the program and all available options (e.g., taking the limits of certain variables), run `python fpt.py --help`. You may consult examples of how to use `fpt.py` in the `examples` folder.

## Examples

We provide the following examples in the `examples` folder:

- `MP.py`: Script to test computation of fixed-point equations using free probability theory for Marchenko-Pastur (MP) law.
- `anisotropic-MP.py`: Script to test computation of fixed-point equations using free probability theory for anisotropic MP law.
- `random-features.py`: Script to test computation of fixed-point equations using free probability theory for generalization error of random features model.
- `ridge-regression-error.py`: Script to test computation of fixed-point equations using free probability theory for generalization error of ridge regression.
- `subordination.py`: Script to test computation of fixed-point equations using free probability theory for subordination.

## auto-fpt Preprint

We overview the algorithms underlying `auto-fpt` and document the various techniques we leverage to make the computation of fixed-point equations more tractable, such as matrix scalarization, sparse block matrix inversion, and duplicate equation identification and pruning. Please find the preprint [here](https://arxiv.org/pdf/2504.10754). You may cite your usage of this library using the arXiv-generated bib entry.
