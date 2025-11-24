'

Forked from [here](https://github.com/xiaochus/TrafficFlowPrediction)

# Traffic flow Prediction System

Authors:

- Hoàng Minh Đức [105541452@student.swin.edu.au](mailto:105541452@student.swin.edu.au)

Running guide:

- Create a Python virtual environment in this directory: `python -m venv .`
- Activate the virtual environment: `./Scripts/Activate.ps1` (Windows) or `source bin/activate` (Linux)
- Install packages: `pip install -r requirements.txt`
- Run `python main.py`. In the output, there is a line that says "Running on public URL", follow the link of that line.
  - or enter `http://127.0.0.1:7860/` if that link works.
- Enjoy!

# Kaggle Notebooks

Note: There's a Kaggle notebook if you want to see the training process, since my laptop doesn't have GPU support. Pretrained weights `/model/*.keras` were included if you want model local inference.

Kaggle notebooks:

- [trainer](https://www.kaggle.com/code/w2nrp2tfb/trainer), used for demonstration.
- [trainer 2](https://www.kaggle.com/code/w2nrp2tfb/trainer-2), more similar to the actual code.

They use the data provided by `boroondara2006.zip`, or the `data/` folder. For convenience, they are also downloaded here.

# Other Links:

- Report: https://docs.google.com/document/d/1RhNKwta_dEkDoBVj0KPk6JPTmrcMJei_wwDdkJebqOs/edit?usp=sharing
- Slides: https://docs.google.com/presentation/d/1JdORTSHtRyzWRiWo-Z6-4UgOW9tXkSqf/edit?usp=sharing&ouid=110314531860417001009&rtpof=true&sd=true
