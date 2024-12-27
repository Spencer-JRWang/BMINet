# Introduction of BMINet
![framework](https://raw.githubusercontent.com/Spencer-JRWang/BMINet/main/Example/image.png)
Machine learning and network based tool for:
- Classify and predict different disease stages precisely
- Understand machenism using SHAP model explanation
- Detect Bone-Muscle Interact network and detect disease modules
# Install
## Dependency
- **Python>=3.9**
## Quick install
- Install from pypi: `pip install BMINet`
- Or install from GitHub: 
```
git clone https://github.com/Spencer-JRWang/BMINet
cd BMINet
pip install .
```
# Example Usage
- First, you should prepare your data (from CT or MRI etc.): [Example Data](https://github.com/Spencer-JRWang/BMINet/blob/main/Example/data.txt)
- Load your data
```python
import pandas as pd
df = pd.read_csv('Example/data.txt', sep='\t')
```
- Select features
```python
# Load FeatureSelector
from BMINet.Interaction import FeatureSelector
selector = FeatureSelector(core_name="LightGBM")
# Conduct feature selection from df
selected_features = selector.select(df)
```
> core_name should be "LightGBM", "XGBoost" or "CatBoost"

- Build linear formula model
```python
from BMINet.Model import Lasso_Formula
formula = Lasso_Formula(df, disease_pairs=("A", "B"))
print(formula)
```
> Also Ridge, ElasticNet and Simple Linear

- Build stacking model
```python
from BMINet.Interaction import StackingModel
# Load stacking model framework
'''
default: StackingModel(
    base_models = [
            ('LGBM', LGBMClassifier(verbose=-1, n_estimators=1000, max_depth=5)),
            ('XGBoost', XGBClassifier(n_estimators=1000, max_depth=5)),
            ('CatBoost', CatBoostClassifier(verbose=False, iterations=800, max_depth=5))
        ]), 
    meta_model = LogisticRegression()
'''
Model = StackingModel()
best_scores = Model.stacking_model_search(df, selected_features)
```
> Using default base model and meta model, you can also define it yourself

- Predict of each group
```python
# single predict
single_prediction_score = Model.single_predict("A vs B", [...], use_our_model=True)
# multiple predict
multiple_prediction_score = Model.multiple_predict_predict("A vs B", [[...], [...],], use_our_model=True)
```
> If you use `use_our model = True`, you are predicting disease staged based on our data and model
> 
> If you are researching on a brand new project, use `use_our_model = False`

- Basic machine learning plots
```python
from BMINet.plot import plot_ml_roc
plot_ml_roc(best_scores)
from BMINet.plot import plot_precision_recall
plot_precision_recall(best_scores)
from BMINet.plot import plot_score_histogram
plot_score_histogram(best_scores)
from BMINet.plot import plot_calibration_curve
plot_calibration_curve(best_scores)
```
- Model Explanation
```python
from BMINet.Interaction import SHAPVisualizer
# Load explanation class and train it
shap_visualizer = SHAPVisualizer(core_name="LightGBM")
shap_visualizer.train_model(df, selected_features)
# The dir you want to save the files
shap_visualizer.plot_shap('./Example')
shap_visualizer.plot_dependence('./Example')
```

- Network Construction
```python
# Load NetworkConstructor
from BMINet.Interaction import NetworkConstructor
network_constructor = NetworkConstructor(core_name="LightGBM", cutoff = 1.5)

# Construct sub-network list for each group
interactions = network_constructor.construct_network(df, selected_features)
# Construct conmbined network
combined_graph = network_constructor.compose_all(interactions)
# Remove isolated nodes from the network
Graph_BMI = network_constructor.remove_isolated_nodes(combined_graph)
# Save to .graphml file
network_constructor.save_graph(Graph_BMI, './Example')
```

- Network Analysis
```python
from BMINet.Interaction import NetworkMetrics
metrics_calculator = NetworkMetrics(Graph_BMI)
metrics = metrics_calculator.compute_metrics()
```

> You can see more Example usage at [here](https://github.com/Spencer-JRWang/BMINet/blob/main/Example.ipynb)

# Update Log
- 2024/9/5: Version `0.1.0`, fix CatBoost bug
- 2024/9/5: Version `0.1.1`, update prediction threholds
- 2024/11/8: Version `0.1.2`, update linear formula model
- 2024/11/26: Version `0.1.3`, add matrix visulization

> BMINet is developed by Jingran Wang

> Department of Orthopedic Surgery, The First Affiliated Hospital of Soochow University

> MOE Key Laboratory of Geriatric Diseases and Immunology, Suzhou Key Laboratory of Pathogen Bioscience and Anti-Infective Medicine
---
<div style="display: flex; justify-content: space-between; align-items: center;">
    <img src="https://raw.githubusercontent.com/Spencer-JRWang/BMINet/main/Example/LOGO.png" alt="Organization Logo" style="width: 100px; height: auto; margin-left: 20px;">
</div>