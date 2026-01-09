from typing import Dict, Any, List
from deepeval import evaluate
from deepeval.metrics import ArgumentCorrectnessMetric, ToolCorrectnessMetric, PlanAdherenceMetric, PlanQualityMetric, StepEfficiencyMetric, TaskCompletionMetric
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.dataset import Golden, EvaluationDataset
from deepeval.tracing import observe, update_current_trace
from collections import defaultdict


class DeepevalManager():
    def __init__(self, evaluation_model: str, evaluation_threshold: str, evaluation_metrics: List):
        """
        DeepevalManager is a unified wrapper class for managing and configuring
        multiple DeepEval evaluation metrics used in LLM-based application
        and tool-augmented agent evaluation.

        This class initializes a predefined set of evaluation metrics with a
        shared evaluation model and threshold, and categorizes each metric
        by its applicable evaluation type.

        Attributes:
            evaluation_model (str):
                The LLM model name used internally by DeepEval metrics
                for evaluation (e.g., "gpt-4o", "gpt-4.1").

            evaluation_threshold (float):
                The pass/fail threshold applied to metrics that support
                threshold-based scoring.

            metrics (dict):
                A dictionary mapping metric names to their configurations.
                Each entry contains:
                    - "metric": an instantiated DeepEval metric object
                    - "eval_type": the expected test case type
                    (e.g., "application", "ToolLLMTestCase")

                Example structure:
                {
                    "planAdherenceMetric": {
                        "metric": PlanAdherenceMetric(...),
                        "eval_type": "application"
                    },
                    "Argument Correctness": {
                        "metric": ArgumentCorrectnessMetric(...),
                        "eval_type": "ToolLLMTestCase"
                    }
                }

        Supported Evaluation Types:
            - application:
                Metrics for evaluating high-level agent behavior such as
                planning, step efficiency, and task completion.

            - ToolLLMTestCase:
                Metrics for evaluating tool usage correctness and
                argument validity in tool-calling LLM scenarios.

        Purpose:
            - Centralize DeepEval metric configuration
            - Ensure consistent model and threshold usage
            - Simplify metric selection based on evaluation context
            (application-level vs tool-level evaluation)
        """

        self.evaluation_model = evaluation_model
        self.evaluation_threshold = evaluation_threshold
        self.metrics = {
            "planAdherenceMetric":{
                "metric": PlanAdherenceMetric(
                    threshold=evaluation_threshold,
                    model=evaluation_model,
                    include_reason=True
                ),
                "eval_type": "application"
            },
            "planQualityMetric": {
                "metric": PlanQualityMetric(
                    threshold=evaluation_threshold,
                    model=evaluation_model,
                    include_reason=True
                ),
                "eval_type": "application"
            },
            "stepEfficiencyMetric": {
                "metric": StepEfficiencyMetric(
                    threshold=evaluation_threshold,
                    model=evaluation_model,
                    include_reason=True
                ),
                "eval_type": "application"
            },
            "taskCompletionMetric":{
                "metric": TaskCompletionMetric(
                    threshold=evaluation_threshold,
                    model=evaluation_model,
                    include_reason=True
                ),
                "eval_type": "application"
            },
            "Argument Correctness": {
                "metric": ArgumentCorrectnessMetric(
                    threshold=evaluation_threshold,
                    model=evaluation_model,
                    include_reason=True
                ),
                "eval_type": "ToolLLMTestCase"
            },
            "Tool Correctness":  {
                "metric": ToolCorrectnessMetric(),
                "eval_type": "ToolLLMTestCase"
            },
        }

        return

    @observe
    def tool_call(self, tools: List[ToolCall]) -> List[ToolCall]:
        """
        Observe and record tool call information used by the agent.

        This method serves as a lightweight wrapper to capture tool call
        metadata for tracing and evaluation purposes. It does not modify
        the tool calls and simply returns them as-is.

        Args:
            tools (List[ToolCall]):
                A list of tool call objects invoked by the agent.

        Returns:
            List[ToolCall]:
                The same list of tool calls, unchanged.

        Notes:
            - Decorated with @observe to enable automatic trace logging.
            - Intended to be called within the agent execution flow
              before updating the main trace.
        """
        return tools

    @observe
    def agent(self, input: str, tools: List[ToolCall], actual_output) -> None:
        """
        Observe and record a single agent execution without returning output.

        This method acts as a trace-only wrapper for agent execution.
        Instead of producing or returning a response, it captures the
        agent's input, tool calls, and final output and stores them in
        the current trace for evaluation and analysis.

        Args:
            input (str):
                The input prompt provided to the agent.

            tools (List[ToolCall]):
                A list of tool calls invoked or used by the agent during
                execution.

            actual_output:
                The final output of the agent.
                Currently provided externally (e.g., from dataset or logs)
                and used only for tracing and evaluation.

        Returns:
            None

        Side Effects:
            - Updates the active trace via `update_current_trace` with:
                - input
                - output
                - tools_called
            - Enables downstream metric evaluation (e.g., DeepEval).

        Notes:
            - Decorated with @observe for automatic observability.
            - This method is intended for offline evaluation or replay
              scenarios rather than live agent inference.
            - Replace `actual_output` with real agent output generation
              when integrating with a live agent pipeline.
        """
        tools = self.tool_call(tools)
        output = actual_output  # TODO output from data
        update_current_trace(
            input=input,
            output=output,
            tools_called=tools
        )


    def evaluate_application(self, eval_data_list: List[Any]) -> Dict:
        """
        Run application-level evaluation using DeepEval metrics.

        This method converts raw evaluation data into DeepEval `Golden`
        objects, constructs an `EvaluationDataset`, and evaluates each
        sample using application-level metrics such as planning quality,
        task completion, and step efficiency.

        The evaluation flow is as follows:
            1. Parse evaluation data and build Golden objects
            2. Create an EvaluationDataset from the goldens
            3. Replay agent execution to populate traces
            4. Compute metric scores from the recorded traces

        Args:
            eval_data_list (List[Any]):
                A list of evaluation samples.
                Each sample is expected to contain:
                    - "task": the input prompt
                    - "시스템_response": the agent's final response
                    - "step": a list of execution steps, each containing
                      tool usage information

        Returns:
            dict:
                A dictionary mapping application-level metric names
                to a list of computed scores.

                Example:
                {
                    "planAdherenceMetric": [0.92],
                    "taskCompletionMetric": [1.0],
                    ...
                }

        Notes:
            - Only metrics with eval_type == "application" are evaluated.
            - Tool calls are reconstructed from step-level tool usage and
              attached to each Golden for trace-based evaluation.
            - This method performs offline evaluation by replaying agent
              executions rather than running live inference.
        """
        
        results = defaultdict(list)

        goldens = []
        for eval_data in eval_data_list:
            input = eval_data["task"]
            response =  eval_data["시스템_response"]
            tools_called = [
                ToolCall(name=tool["function"]["name"])
                for step in eval_data["step"]
                for tool in step["tool"]
            ]
            
            goldens.append(Golden(input=input, tools_called=tools_called, actual_output=response))

        dataset = EvaluationDataset(goldens=goldens)
        target_metrics = [
            value["metric"] for key, value in self.metrics.items() 
            if value["eval_type"] == "application"
        ]
        
        for golden in dataset.evals_iterator(metrics=target_metrics):
            self.agent(golden.input, golden.tools_called, golden.actual_output)
        
        for metric_name, metric_value in self.metrics.items():
            if metric_value['eval_type'] == 'application':
                results[metric_name].append(metric_value["metric"].score)

        return results
