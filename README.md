# Setup

## Python Environment

Environment is using conda. To install conda, follow the instructions [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).
You can then create a new environment using the following command:

```bash
conda env create --prefix ./.conda --file environment.yaml
```

where `<env>` is the name of the environment you want to create. You can then activate the environment using:

```bash
conda activate ./.conda
```

To export the environment (to add new packages), use:

```bash
conda env export --file environment.yaml
```

## Notebooks

The notebooks are located in the `notebooks` directory:

- `notebooks/h1st_model.ipynb`: My version of https://www.kaggle.com/code/wadjihbencheikh/mod-le-1-ms-azure. It requires csv files from [Kaggle](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)
- `notebooks/Aitomatic.ipynb`: Final solution, using provided data files.

## Training code

Main executables are `train.py` and `predict.py`. In future, they would include CLI and API. Right now, they are just executable scripts.
The model logic is defined in `predictive_maintenance/feature_transformers.py` and `predictive_maintenance/modeler.py`.
They make use of the generic implementation of XGBoost model (as part of H1st framework) is in `ml_model/ml_xgboost.py`.

## Application

`predict.py` is but a demo/test script. The real application is in `streamlit_app.py`. It is a Streamlit app that uses the same model as `predict.py` to predict machine failures and visualize the results.
To run the app locally, use:

```bash
streamlit run streamlit_app.py
```

Streamlit cloud will automatically updat the app at <https://marov-predictive-maintenance-model-streamlit-app-0mbk0u.streamlit.app> when you push to the repo.
