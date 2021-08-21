# Summary of the snippets included in this folder

## print_metrics.py
Print <span style='color:green;'>color</span>-<span style='color:tomato;'>coded</span> TensorFlow metrics in IPython 
    for Training, Validation and Test data or datasets in <span style='border: 1px solid; padding: 1px 3px;'>tabular format</span>, 
    comparing those with the another metrics (such as previous history), if any, e.g. 

![Metrics Compared](media/MetricsTableCompared.jpg)

Printing validation and test metrics is optional.<br>
While validation metrics are read from history objects, test metrics are evaluated on the fly (if models are provided).

The metrics can be variable in number.<br>
If you are comparing metrics, it's good to have the same types of metrics in previous and current histories.<br>
This snippet will work in any case, but what it shows may not be correct if the histories provided are incorrect.

### Signature
```python
def print_metrics(history, epochs, model=None, *, ds_test=None, X_test=None, y_test=None, 
                  prev_hist=None, prev_model=None, prev_epochs="", 
                  metric_names=None, ret_metrics=True):
```
### Parameters

||||
|-|-|-|
| history | dict | Typically TensorFlow's history.history (having keys such as loss, accuracy, lr etc.)|
| epochs | int |
| model | Model| This will be used to get metrics_names (if not explicitly provided), and for evaluating test data.
| ds_test | Dataset | Provide this (or X_test, y_test below) if you want to see test metrics.
| X_test | Any data iterator that your model will accept
| y_test | Any data iterator that your model will accept
| prev_hist | pandas DataFrame (e.g. read from a csv file)
| prev_model | Model | Provide this for evaluating test data and comparing against the results from new model.
| prev_epochs | int
| metric_names | list of strings (default: model.metrics_names)
| ret_metrics | Boolean | True: if evaluate is called on model, metrics generated will be returned by the function as a dict of metrics names and their values.

<hr>