'
Forked from [here](https://github.com/xiaochus/TrafficFlowPrediction)

How to setup:

- clone this repo. Use Python 3.13 (or whatever, 3.13 is preferred)
- create a venv: `python -m venv .`
- Activate the venv: `./Scripts/Activate.ps1`
- Install packages: `pip install -r requirements.txt`

Authors:

- Hoàng Minh Đức [105541452@student.swin.edu.au](mailto:105541452@student.swin.edu.au)

Running guide:

- Create a Python virtual environment in this directory: `python -m venv .`
- Activate the virtual environment: `./Scripts/Activate.ps1` (Windows) or `source bin/activate` (Linux)
- Install packages: `pip install -r requirements.txt`
- Run `python main.py`. In the output, there is a line that says "Running on public URL", follow the link of that line.
  - or enter `http://127.0.0.1:7860/` if that link works.
- Enjoy!

Note: There's a Kaggle notebook if you want to see the training process, since my laptop doesn't have GPU support. Pretrained weights `/model/*.keras` were included.
