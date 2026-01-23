from evalscope import TaskConfig, run_task
from evalscope.constants import EvalType, JudgeStrategy

task_cfg = TaskConfig(
    model='gpt-4.1',
    api_url='https://api.openai.com/v1',
    api_key='sk-###',
    eval_type=EvalType.SERVICE,
    datasets=[
        'omni_doc_bench', 
    ],

    eval_batch_size=8,
    generation_config={
        'max_tokens': 32768,  
        'temperature': 0.7,  
        'presence_penalty': 1.5, 
        'n': 1,  
    },
    limit=5,
)

run_task(task_cfg=task_cfg)