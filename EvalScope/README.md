
### Set environment
```
$ conda create -n evalscope_env python==3.10
$ conda activate evalscope_env
$ (evalscope_env) cd EvalScope
$ (evalscope_env) pip install -r requirements.txt
```

### Run evaluation
```
$ (evalscope_env) cd EvalScope
$ (evalscope_env) python evaluation.py
```

### View visualization
```
$ (evalscope_env) evalscope app # 0.0.0.0:7860
```