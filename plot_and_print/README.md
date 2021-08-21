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

<hr>