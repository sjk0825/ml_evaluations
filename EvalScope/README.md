
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
$ (evalscope_env) python evaluation_mQA.py
$ (evalscope_env) python evaluation_retrieval.py
```

### View visualization
```
$ (evalscope_env) evalscope app --lang en  --server-port 5000 # 0.0.0.0:7860

```
<img width="1462" height="925" alt="image" src="https://github.com/user-attachments/assets/6666e45b-c3b9-4fc6-a2ce-956b379e6068" />
