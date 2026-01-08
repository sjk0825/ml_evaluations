import os
from typing import List, Dict, Callable
from contextlib import contextmanager

import mlflow
from mlflow.genai import scorer
from mlflow.genai.scorers import Correctness, Guidelines


class MLflowLogger():
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str,
        run_name: str = None,
        tags: Dict = None,
        scorers: List = None
    ):
        """
        MLflowLogger 클래스 초기화.

        MLflow 실험(experiment)을 설정하고, 필요 시 추적 URI(tracking URI)를 지정합니다.
        또한 각 실행(run)에 대한 이름과 태그(tags)과 사용자가 지정한 score 리스트를 저장합니다.

        Args:
            experiment_name (str): MLflow 실험 이름.
            tracking_uri (str): MLflow tracking 서버 URI. 기본값은 None.
            run_name (str, optional): MLflow 실행(run) 이름. 기본값은 None.
            tags (dict, optional): MLflow 실행(run)에 추가할 태그. 기본값은 None.
            scorers (List, optional): MLflow llmjudge 평가(evaluate) metric 리스트. 기본값은 None.
        """

        # set connection
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)

        mlflow.set_experiment(experiment_name)

        self.run_name = run_name
        self.tags = tags or {}

        # set evaluation
        scorers_registry = {
            "CORRECTNESS": Correctness(),
            "IS_CONCISE": is_concise, 
            "IS_KOREAN": Guidelines(name="is_korean", guidelines="The answer must be in 한글")
        }

        if scorers is not None:
            self.scorers = []
            for scorer in scorers:
                self.scorers.append(scorers_registry[scorer])


    @contextmanager
    def start_run(self):
        """
        MLflow 실행(run)을 시작하고 컨텍스트 관리자로 사용.

        사용 예시:
            with logger.start_run():
                logger.log_params({"lr": 0.001})
                logger.log_metrics({"loss": 0.5})

        실행(run) 시작 시, 설정된 태그(tags)를 자동으로 적용합니다.

        Yields:
            None
        """
        with mlflow.start_run(run_name=self.run_name):
            mlflow.set_tags(self.tags)
            yield

    def log_params(self, params: dict) -> None:
        """
        MLflow에 파라미터를 기록합니다.

        Args:
            params (dict): 기록할 파라미터 딕셔너리. 
                           키(key)는 파라미터 이름, 값(value)은 파라미터 값.
        """
        if params:
            mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int = None) -> None:
        """
        MLflow에 메트릭(metric)을 기록합니다.

        Args:
            metrics (dict): 기록할 메트릭 딕셔너리.
                            키(key)는 메트릭 이름, 값(value)은 int 또는 float 값.
            step (int, optional): 메트릭 기록 시 step 값. 기본값은 None.
        """
        if not metrics:
            return

        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(k, v, step=step)
    
    def evaluate(self, dataset: List, answer_generator: Callable) -> None:
        """
        Evaluate a generative QA model using MLflow GenAI evaluation.

        Args:
            dataset (List[Dict[str, Any]]):
                Evaluation dataset. Each item should contain at least
                the model input and reference fields expected by the scorers.
            answer_generator (Callable):
                A prediction function that takes one data sample and
                returns a generated answer string.

        """

        results = mlflow.genai.evaluate(
            data=dataset,
            predict_fn=answer_generator,
            scorers=self.scorers,
        )
    
@scorer
def is_concise(outputs: str) -> bool:
    """Evaluate if the answer is concise (less than 5 words)"""
    return len(outputs.split()) <= 10
