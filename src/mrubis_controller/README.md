## Setup development environment

Prerequisites:
- `tmux` installed

Create conda environment with python version >= 3.9
```
conda create --name marl python=3.9 ipython
conda activate marl
```

Install dependencies
```
pip install -r requirements.txt
```

Place the `mRubis.jar` file at the following location: `../mRUBiS/ML_based_Control/`

## Training

Run `main.py`:
```
python3 main.py --runner
```