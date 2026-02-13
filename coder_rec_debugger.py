from __future__ import annotations

import logging
from rich.console import Console

from agents import Agent, Runner
from agents.tracing import gen_trace_id, trace
from agents.model_settings import ModelSettings
from black import format_str, Mode

from database import Program
from utils.code import apply_diff, parse_evolve_blocks
from utils.datatypes import IdeaData, reasoning_models
from utils.format import format_metrics_safe

logger = logging.getLogger(__name__)

console = Console()

DEBUGGER_INSTRUCTIONS = """You are an expert developer and researcher who ensures modified code runs correctly and properly implements research ideas.
Your task is to analyze code, identify any kind of errors, including syntax errors, runtime errors, or logical issues, and verify functionality.
Provide detailed diagnostics and specific fixes when problems are found.
Consider edge cases and ensure the code fully addresses the research requirements.

You MUST use the exact SEARCH/REPLACE diff format. Do NOT use Git diff format. Do NOT use line prefixes like `+`, `-`, or `@@`.

Use this structure exactly:
```
<<<<<<< SEARCH
# Code with error (must match exactly)
=======
# DEBUG: <comment>
# Fixed code here
>>>>>>> REPLACE
```
Example 1 for debugging a syntax error:
```
<<<<<<< SEARCH
def compute_mean(values):
    total = sum(values
    return total / len(values)
=======
def compute_mean(values):
    # DEBUG: missing parenthesis in function call, fixed by adding parenthesis
    total = sum(values)
    return total / len(values)
>>>>>>> REPLACE
```

Use Comments like `# DEBUG: <comment>` to indicate the changes you made when debugging.
"""



DEBUGGER_TEMPLATE = """
Resolve the following error in a multi-file Python codebase.

An error occurred during execution:
```
{error_message}
```

Below is the code that caused the error:
```{language}
{modified_code}
````

The `main_code.py` and `models.py` modification was made to implement the idea:
```json
{idea}
```
```

The `diagnosis_tools.py` modification was made to implement the idea:
```json
{idea_diagnosis}
```

Your responsibilities:

- Identify and fix the cause of the error in the modified code.
- Ensure that all involved files and components integrate correctly and run without errors.
- Ensure the code modification do not break the research idea.
- Ensure the new code within the `Self_EvolveRec` block is reachable in the workflow. New code should be executed as new idea but not suppressed by error handling or cheated by None values.
- If necessary, update function inputs or implementations to ensure consistency.
- If the code depends on a library that is not available, use the standard library instead.

Please analyze the error and return the corrected code using `diff` format.
"""

class Debugger:
    def __init__(self, developer: str, debugger: str, reasoning_effort: str = 'medium'):
        
        self.debugger = Agent(
            name="Code debugging agent",
            instructions=DEBUGGER_INSTRUCTIONS,
            model=debugger,
            model_settings=ModelSettings(reasoning={'effort': reasoning_effort}) if debugger in reasoning_models else ModelSettings(),
            output_type=str,
        )

        self.query = None
        self.problem_description = None
        self.language = None
        self.trace_id = None
        self.problem_name = 'NA'

    def update_topic(self, query: str, problem_name: str, problem_description: str):
        self.query = query
        self.problem_name = problem_name
        self.problem_description = problem_description

    async def debug(
        self, input_code: str, error_message: str,
    ) -> str:
        trace_id = self.trace_id
        if trace_id is None:
            trace_id = gen_trace_id()
            self.trace_id = trace_id

        with trace(f"Self_EvolveRec_{self.problem_name}", trace_id=trace_id, disabled=False):
            debugger_input = DEBUGGER_TEMPLATE.format(
                # query=self.query,
                error_message=error_message,
                modified_code=input_code,
                idea=self.idea.model_dump(),
                idea_diagnosis=self.idea_diagnosis.model_dump(),
                language=self.language,
            )
            result = await Runner.run(self.debugger, debugger_input)

            logger.info(f"Debugger error message:\n {error_message}")
            logger.info(f"Debugger changes:\n {result.final_output_as(str)}")

            diff_with_text = result.final_output_as(str)
            output_code = apply_diff(input_code, diff_with_text)
            
            try:
                output_code = format_str(output_code, mode=Mode())
            except Exception as e:
                logger.warning(f"Error when formatting code: {e}")
                pass
            return output_code