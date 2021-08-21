# Author        : manisar (manisar2 on github)
# License       : MIT License
# Location      : https://github.com/manisar2/MLSnippets/blob/main/plot_and_print/print_metrics.py
# Description   : Check https://github.com/manisar2/MLSnippets/tree/main/plot_and_print

# Note: For displaying the HTML table in IPython, I use the git package ipyccmd that can be installed as:
    # pip install git+https://github.com/manisar2/ipyccmd.git
    # This is used at only couple of places - in the end.
    # You may avoid it by using these two lines:
    # from IPython.display import display_html
    # display_html(MetricsTable)

from ipyccmd import DisplayType

def print_metrics(history, epochs, model=None, *, ds_test=None, X_test=None, y_test=None, 
                  prev_hist=None, prev_model=None, prev_epochs="", 
                  metric_names=None, ret_metrics=True):
    """
    Print color-coded TensorFlow metrics in IPython for Training, Validation and Test data or datasets in tabular \
    format, comparing those with the another (previous, e.g.) metrics (if any).

    Note: If using TensorFlow's CSVLogger for saving and loading previous history, be aware \
        that in append mode, it doesn't change headings. E.g. if lr callback \
        was only used in newer run, it's values get inserted into the csv file - so now we have \
        one extra item per row while the column headings are the same as before!
    Parameters
    ----------
    history : dict
        Typically TensorFlow's history.history (having keys such as loss, accuracy, lr etc.)
    epochs : int
    model : Model
        This will be used to get metrics_names (if not explicitly provided), and for evaluating \
        test data.
    ds_test: Dataset
        Provide this (or X_test, y_test below) if you want to see test metrics.
    X_test : Any data iterator that your model will accept
    y_test : Any data iterator that your model will accept
    prev_hist : pandas DataFrame (e.g. read from a csv file)
    prev_model : Model
        Provide this for evaluating test data and comparing against the results from new model.
    prev_epochs : int
    metric_names : list of strings (default: model.metrics_names)
    ret_metrics : Boolean
        True: if evaluate is called on model, metrics generated will be returned by the function \
            as a dict of metrics names and their values.
    """
    "---".md()
    if metric_names is None and not model:
        raise ValueError("metric_names needs to be set if no model is provided")
    if metric_names is None: metric_names = model.metrics_names
    is_test_data_available = False
    if ds_test is not None or (X_test is not None and y_test is not None): is_test_data_available = True
    prev_epochs = "" if prev_epochs == 0 else prev_epochs
    prev_lr = ""
    curr_lr = ""
    prev_tr_metrics, prev_tr_val_metrics, prev_te_metrics, \
        curr_tr_metrics, curr_tr_val_metrics, curr_te_metrics = {}, {}, {}, {}, {}, {}

    validation_present = False
    if len([k for k in history.keys() if k.startswith("val")]) > 0: validation_present = True

    # Previous training metrics
    if prev_hist is not None: # prev_hist can be None as well if exception while loading .csv
        if "lr" in prev_hist.keys(): prev_lr = f"{prev_hist.lr.iat[-1]:.10f}"
        for metric in metric_names:
            val_metric = "val_" + metric
            if prev_hist.get(metric) is not None:
                prev_tr_metrics[metric] = round(prev_hist[metric].iat[-1], 10)
                if "accuracy" in metric: prev_tr_metrics[metric] = f"{100*prev_tr_metrics[metric]:.2f}%"
                if validation_present and prev_hist.get(val_metric) is not None:
                    prev_tr_val_metrics[val_metric] = round(prev_hist[val_metric].iat[-1], 10)
                    if "accuracy" in metric: prev_tr_val_metrics[val_metric] = f"{100*prev_tr_val_metrics[val_metric]:.2f}%"
    
    # Previous test metrics
    prev_eval_metrics = []
    if prev_model and is_test_data_available:
        if ds_test: prev_eval_metrics = prev_model.evaluate(ds_test, verbose=0)
        else: prev_eval_metrics = prev_model.evaluate(X_test, y_test, verbose=0)
        for metric in metric_names:
            model_metric_names = prev_model.metrics_names
            indx = model_metric_names.index(metric) if metric in model_metric_names else -1
            if indx != -1: 
                prev_te_metrics[metric] = round(prev_eval_metrics[indx], 10)
                if "accuracy" in metric: 
                    prev_te_metrics[metric] = f"{100*prev_te_metrics[metric]:.2f}%"
    ################################################################################    

    # Current training metrics
    curr_lr = f"{model.optimizer.lr.numpy():.10f}"
    if history is not None:
        # if "lr" in history.keys(): prev_lr = f"{history['lr'].iat[-1]:.10f}"
        for metric in metric_names:
            val_metric = "val_" + metric
            curr_tr_metrics[metric] = round(history[metric][-1], 10)
            if not prev_tr_metrics.get(metric): prev_tr_metrics[metric] = ""
            if validation_present:
                curr_tr_val_metrics[val_metric] = round(history[val_metric][-1], 10)
                if not prev_tr_val_metrics.get(val_metric): prev_tr_val_metrics[val_metric] = ""
            if "accuracy" in metric:
                curr_tr_metrics[metric] = f"{100*curr_tr_metrics[metric]:.2f}%"
                if validation_present: curr_tr_val_metrics[val_metric] = f"{100*curr_tr_val_metrics[val_metric]:.2f}%"

    # Current test metrics
    if model and is_test_data_available:
        if ds_test: metrics = model.evaluate(ds_test, verbose=0)
        else: metrics = model.evaluate(X_test, y_test, verbose=0)
        for metric in metric_names:
            model_metric_names = model.metrics_names
            indx = model_metric_names.index(metric) if metric in model_metric_names else -1
            if indx != -1:
                curr_te_metrics[metric] = round(metrics[indx], 10)
                # if not prev_te_metrics.get(metric): prev_te_metrics[metric] = "" # done later
                if "accuracy" in metric: 
                    curr_te_metrics[metric] = f"{100*curr_te_metrics[metric]:.2f}%"
    else: metrics = [None] * len(metric_names) # to be used for returning if needed
    ################################################################################

    # Adjusting extra metrics from prev_hist (if metrics changed)
    if prev_hist is not None:
        prev_tr_keys = set(prev_hist.keys())
        curr_tr_keys = set(curr_tr_metrics.keys())
        extra_prev_tr_keys = prev_tr_keys - curr_tr_keys
        for key in extra_prev_tr_keys:
            if key.startswith("val_"): continue
            metric_names.append(key)
            prev_tr_metrics[key] = round(prev_hist[key].iat[-1], 10)
            if "accuracy" in key: prev_tr_metrics[key] = f"{100*prev_tr_metrics[key]:.2f}%"
            curr_tr_metrics[key] = ""
            val_key = "val_" + key
            if prev_hist.get(val_key) is not None: 
                prev_tr_val_metrics[val_key] = round(prev_hist[val_key].iat[-1], 10)
                if "accuracy" in key: prev_tr_val_metrics[val_key] = f"{100*prev_tr_val_metrics[val_key]:.2f}%"
                curr_tr_val_metrics[val_key] = ""

            if prev_model and is_test_data_available:
                model_metric_names = prev_model.metrics_names
                indx = model_metric_names.index(key) if key in model_metric_names else -1
                if indx != -1:
                    prev_te_metrics[key] = round(prev_eval_metrics[indx], 10)
                    if "accuracy" in key: prev_te_metrics[key] = f"{100*prev_te_metrics[key]:.2f}%"
                    curr_te_metrics[key] = ""

    # Adjusting test metrics differences (if test wasn't done in prev or curr)
    prev_te_keys = set(prev_te_metrics.keys())
    curr_te_keys = set(curr_te_metrics.keys())
    for key in prev_te_keys - curr_te_keys: curr_te_metrics[key] = ""
    for key in curr_te_keys - prev_te_keys: prev_te_metrics[key] = ""
    for key in set(metric_names) - set(curr_te_metrics.keys()):
        curr_te_metrics[key] = ""
        prev_te_metrics[key] = ""
    ################################################################################

    def get_tdstyle(prev_metric, curr_metric, higher_is_better=True):
        tdstyle = ""
        if prev_metric == "" or curr_metric == "": return tdstyle
        good_style = " style='color: green; font-weight:bold;'"
        bad_style = " style='color: tomato;'"
        if type(prev_metric) is str: prev_metric = float(prev_metric.strip("%"))
        else: prev_metric = float(prev_metric)
        if type(curr_metric) is str: curr_metric = float(curr_metric.strip("%"))
        else: curr_metric = float(curr_metric)
        result = curr_metric - prev_metric
        if higher_is_better:
            if result > 0: tdstyle = good_style
            if result < 0: tdstyle = bad_style
        else: 
            if result > 0: tdstyle = bad_style
            if result < 0: tdstyle = good_style
        return tdstyle

    len_metrics = len(metric_names)
    nr_training = len_metrics + 3
    nr_vte = len_metrics + 1

    thtml = f"""
        <table style='border: 1px solid;'>
        <thead style='border-bottom: 1px solid;'>
            <tr><th></th><th></th><th>Prev</th><th>Current</th></tr>
        </thead>
        <tbody>
            <tr><td rowspan={nr_training} style='border: 1px solid;'>Training</td></tr>
            <tr style='border-top: 1px solid;'><td>Epochs</td><td>{prev_epochs}</td><td>{epochs}</td></tr>
            <tr><td>LR</td><td>{prev_lr}</td><td>{curr_lr}</td></tr>"""
    for i, metric in enumerate(metric_names): # Training
        tdstyle = ""
        if metric == "loss": tdstyle = get_tdstyle(prev_tr_metrics[metric], curr_tr_metrics[metric], False)
        elif "accuracy" in metric: tdstyle = get_tdstyle(prev_tr_metrics[metric], curr_tr_metrics[metric], True)
        thtml += f"""
        <tr><td>{metric.replace("_", " ").title()}</td><td>{prev_tr_metrics[metric]}</td>
            <td {tdstyle}>{curr_tr_metrics[metric]}</td></tr>"""

    thtml += f"""
        <tr style='border-top: 1px solid;'><td rowspan={nr_vte} style='border: 1px solid;'>Validation</td></tr>
        """
    for i, metric in enumerate(metric_names): # Validation
        bstring = "" if i else " style='border-top: 1px solid;'"
        tdstyle = ""
        val_metric = "val_" + metric
        if metric == "loss": tdstyle = get_tdstyle(prev_tr_val_metrics[val_metric], curr_tr_val_metrics[val_metric], False)
        elif "accuracy" in metric: tdstyle = get_tdstyle(prev_tr_val_metrics[val_metric], curr_tr_val_metrics[val_metric], True)
        thtml += f"""
        <tr{bstring}><td>{metric.replace("_", " ").title()}</td><td>{prev_tr_val_metrics[val_metric]}</td>
            <td{tdstyle}>{curr_tr_val_metrics[val_metric]}</td></tr>"""

    thtml += f"""
        <tr style='border-top: 1px solid;'><td rowspan={nr_vte} style='border: 1px solid;'>Test</td></tr>"""
    for i, metric in enumerate(metric_names): # Test
        bstring = "" if i else " style='border-top: 1px solid;'"
        tdstyle = ""
        if metric == "loss": tdstyle = get_tdstyle(prev_te_metrics[metric], curr_te_metrics[metric], False)
        elif "accuracy" in metric: tdstyle = get_tdstyle(prev_te_metrics[metric], curr_te_metrics[metric], True)
        thtml += f"""
        <tr{bstring}><td>{metric.replace("_", " ").title()}</td><td>{prev_te_metrics[metric]}</td>
            <td{tdstyle}>{curr_te_metrics[metric]}</td></tr>"""    

    MetricsTable = thtml + "</tbody></table>"
    MetricsTable.md(DisplayType.HTML) # md() is from a git package ipyccmd (see the Note at the top)
    "---".md()
    if ret_metrics: return dict(zip(metric_names, metrics))
    # zip will automatically exclude items appended to metric_names later (from prev_hist exclusively)
    ############################################################################
