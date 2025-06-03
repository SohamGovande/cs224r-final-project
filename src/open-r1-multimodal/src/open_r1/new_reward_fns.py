import re
import os
import sys
import ast
import json
import subprocess
import tempfile
from datetime import datetime
from contextlib import contextmanager
from typing import List, Tuple, Dict, Optional, Callable, Any
from Levenshtein import ratio
import traceback
import sys, subprocess, tempfile, os, textwrap

try:
    import signal
    HAS_SIGNAL = True
except ImportError:
    HAS_SIGNAL = False
    print("Warning: signal module not available, timeout functionality disabled")


class PythonCodeExecutor:
    """Execute Python code safely with timeout and capture results."""
    
    @staticmethod
    @contextmanager
    def timeout(seconds):
        """Context manager for timeout. Note: Only works on Unix-like systems."""
        if HAS_SIGNAL and hasattr(signal, 'SIGALRM'):
            def signal_handler(signum, frame):
                raise TimeoutError("Code execution timed out")
            signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)
        else:
            yield
    
    @staticmethod
    def extract_python_code(text: str) -> str:
        """Extract Python code from the response."""
        # First try to extract from answer tags
        answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
        if answer_match:
            text = answer_match.group(1).strip()
        
        # Try to extract code blocks
        code_blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[-1].strip()
        
        code_blocks = re.findall(r'```\n(.*?)```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[-1].strip()
        
        # If no code blocks, return the text itself
        return text.strip()
    
    @staticmethod
    def check_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """Check if Python code has valid syntax."""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
    
    @staticmethod
    def execute_with_test(code: str, test_code: str, timeout: int = 5) -> Dict[str, any]:
        """Execute Python code with test cases."""
        full_code = code + "\n\n" + test_code
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(full_code)
            f.flush()
            
            try:
                result = subprocess.run(
                    [sys.executable, f.name],
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                
                success = result.returncode == 0
                stdout = result.stdout
                stderr = result.stderr
                
                return {
                    'success': success,
                    'stdout': stdout,
                    'stderr': stderr,
                    'error': stderr if stderr else None
                }
            except subprocess.TimeoutExpired:
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': 'Execution timed out',
                    'error': 'Timeout'
                }
            except Exception as e:
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': str(e),
                    'error': str(e)
                }
            finally:
                try:
                    os.unlink(f.name)
                except:
                    pass

def clean_text(text: str, exclude_chars: List[str] = ['\n', '\r']) -> str:
    """Clean and normalize text for comparison."""
    # Extract content between <answer> and </answer> if present
    answer_matches = re.findall(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if answer_matches:
        text = answer_matches[-1]
    
    for char in exclude_chars:
        if char in ['\n', '\r']:
            # If there is a space before the newline, remove the newline
            text = re.sub(r'(?<=\s)' + re.escape(char), '', text)
            # If there is no space before the newline, replace it with a space
            text = re.sub(r'(?<!\s)' + re.escape(char), ' ', text)
        else:
            text = text.replace(char, ' ')
    
    # Remove leading and trailing spaces and convert to lowercase
    return text.strip().rstrip('.').lower()

def __unit_test_reward(prompt: str, candidate_code: str, test_code: str, entry_point: str, **kwargs) -> float:
    """Return 1.0 if all unit tests pass, otherwise 0.0 (adapted from provided script)."""
    candidate_block = "# MODEL COMPLETION\n" + candidate_code.rstrip()
    should_indent = kwargs.get('should_indent', True)
    if should_indent:
        indented_block = textwrap.indent(candidate_block, "    ")
    else:
        indented_block = candidate_block
    
    harness_code = (
        f"# PROMPT\n{prompt}\n"
        f"{indented_block}\n\n"
        f"{test_code}\n\n"
        f"if __name__ == '__main__':\n"
        f"    check({entry_point})\n"
    )

    prompt_name = f"data-json/{entry_point}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(prompt_name, 'w') as f:
        f.write(harness_code)

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(harness_code)
        tmp_path = tmp.name

    try:
        proc = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=5)
        return 1.0 if proc.returncode == 0 else 0.0
    except subprocess.TimeoutExpired:
        return 0.0
    finally:
        os.unlink(tmp_path)

def _unit_test_reward(generated_code: str, function_header: str, test_code: str, entry_point: str, **kwargs) -> float:
    """Wrapper around unit_test_rewards to fit the reward API used elsewhere."""
    if test_code is None:
        # No tests provided – cannot evaluate accuracy.
        return 0.0

    candidate_code = PythonCodeExecutor.extract_python_code(generated_code)
    # prompt_name = f"data/{entry_point}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    # return __unit_test_reward(function_header, candidate_code, test_code, entry_point, **kwargs)
    try:
        # with open(prompt_name, 'w') as f:
        #     f.write(candidate_code)
        json_code = json.loads(candidate_code.strip())
        candidate_code = json_code['code']
        candidate_language = json_code['language']
        if candidate_language != 'python': raise ValueError(f"Candidate language is {candidate_language}, not python")
        if len(candidate_code) < 100: raise ValueError(f"Candidate code is too short: {candidate_code}")
        # return 1
        return 0.5 + 0.5 * __unit_test_reward(function_header, candidate_code, test_code, entry_point, **kwargs)
    except (ValueError, KeyError, json.JSONDecodeError, TypeError, AttributeError) as e:
        print(f"Error parsing generated code: {e}")
        return 0.0
    

    # return __unit_test_reward(function_header, candidate_code, test_code, entry_point, **kwargs)

def _python_syntax_reward(completions: str, **kwargs) -> float:
    return 0

    assert type(completions) == str, f"completions must be a string, but got {type(completions)}"
    """Check if the Python code has valid format."""
    code = PythonCodeExecutor.extract_python_code(completions)
    is_valid, error = PythonCodeExecutor.check_syntax(code)
    
    if kwargs.get('debug', False) and not is_valid:
        print(f"format Error: {error}")
        print(f"Code: {code[:200]}...")
    
    return 1.0 if is_valid else 0.0

def _python_execution_reward(completions: str, **kwargs) -> float:
    assert type(completions) == str, f"completions must be a string, but got {type(completions)}"
    """Check if the Python code executes without errors."""
    code = PythonCodeExecutor.extract_python_code(completions)
    
    # Skip accuracy reward if format is invalid
    syntax_valid, _ = PythonCodeExecutor.check_syntax(code)
    if not syntax_valid:
        return 0.0
    
    # For code snippets, wrap them in a function to test accuracy
    wrapped_code = f"""
def test_execution():
    # Common variables that might be used in HumanEval problems
    numbers = [1, 2, 3, 4, 5]
    threshold = 2
    paren_string = "((hello))"
    arr = [1, 2, 3]
    s = "hello"
    n = 5
    
    # Execute the code snippet
{chr(10).join('    ' + line for line in code.split(chr(10)) if line.strip())}

# Test accuracy
test_execution()
"""
    
    result = PythonCodeExecutor.execute_with_test(wrapped_code, "")
    
    if kwargs.get('debug', False) and not result['success']:
        print(f"accuracy Error: {result['error']}")
        print(f"Stderr: {result['stderr']}")
    
    return 1.0 if result['success'] else 0.0

def python_syntax_reward(prompts: list[str], completions: list[str], **kwargs) -> float:
    assert type(prompts) == list and type(completions) == list, f"prompts and completions must be lists, but got {type(prompts)} and {type(completions)}"
    assert len(prompts) == len(completions), f"prompts and completions must have the same length, but got {len(prompts)} and {len(completions)}"
    rewards = []
    for i in range(len(completions)):
        for j in range(len(completions[i])):
            reward = _python_syntax_reward(completions[i][j]['content'], **kwargs)
            rewards.append(reward)
    return rewards


def python_execution_reward(prompts: list[str], completions: list[str], **kwargs) -> float:
    assert type(prompts) == list and type(completions) == list, f"prompts and completions must be lists, but got {type(prompts)} and {type(completions)}"
    assert len(prompts) == len(completions), f"prompts and completions must have the same length, but got {len(prompts)} and {len(completions)}"
    total_reward = 0.0
    num_rewards = 0
    ids, prompts, test_codes, entry_points, function_headers = kwargs['id'], kwargs['problem'], kwargs['test_code'], kwargs['entry_point'], kwargs['function_header']
    rewards = []

    print('Testing python execution reward')
    for i in range(len(completions)):
        for j in range(len(completions[i])):
            prompt = prompts[i]
            test_code = test_codes[i]
            entry_point = entry_points[i]
            function_header = function_headers[i]
            reward = max(_unit_test_reward(completions[i][j]['content'], function_header, test_code, entry_point, should_indent=True), _unit_test_reward(completions[i][j]['content'], "", test_code, entry_point, should_indent=False))
            rewards.append(reward)
    return rewards


def code_completeness_reward(completions: str, **kwargs) -> float:
    assert type(completions) == str, f"completions must be a string, but got {type(completions)}"
    """Check if the generated code is complete with nuanced scoring."""
    code = PythonCodeExecutor.extract_python_code(completions)
    
    # Skip if format is invalid
    syntax_valid, _ = PythonCodeExecutor.check_syntax(code)
    if not syntax_valid:
        return 0.0
    
    # Check for common placeholder patterns
    placeholder_patterns = [
        r'pass\s*$',
        r'#\s*TODO',
        r'#\s*FIXME',
        r'raise\s+NotImplementedError',
        r'\.\.\.',
        r'# Your code here',
        r'# Implementation goes here',
        r'<FILL_ME>',
        r'# placeholder'
    ]
    
    # Check code length and structure
    code_lines = [line for line in code.split('\n') if line.strip() and not line.strip().startswith('#')]
    line_count = len(code_lines)
    
    # Calculate line completeness penalty
    line_penalty = 0.0
    if line_count < 5:
        line_penalty = (5 - line_count) * 0.1
    
    # Check for implementation keywords
    implementation_score = 0.0
    keywords = ['for', 'while', 'if', 'def', 'return', 'import', 'from']
    keyword_count = sum(1 for keyword in keywords if keyword in code)
    implementation_score = min(keyword_count * 0.1, 0.4)
    
    # Check for function definition
    function_def_score = 0.3 if re.search(r'def\s+\w+\s*\(', code) else 0.0
    
    # Check for placeholders
    placeholder_penalty = 0.0
    has_placeholder = any(re.search(pattern, code, re.IGNORECASE) for pattern in placeholder_patterns)
    if has_placeholder:
        placeholder_penalty = 0.3
    
    # Calculate final reward
    reward = max(0, function_def_score + implementation_score - placeholder_penalty - line_penalty)
    
    if kwargs.get('debug', False):
        print(f"Completeness Reward: {reward:.2f}")
        print(f"Function def: {function_def_score:.1f}, Implementation: {implementation_score:.1f}")
        print(f"Placeholder penalty: -{placeholder_penalty:.1f}, Line penalty: -{line_penalty:.1f}")
    
    return reward

def code_coverage_reward(generated_code: str, expected_code: str = None, test_code: str = None, **kwargs) -> float:
    """Check if code handles edge cases using test cases."""
    # Extract think section
    think_match = re.search(r'<think>(.*?)</think>', generated_code, re.DOTALL)
    think_content = think_match.group(1).lower() if think_match else ""
    
    # Edge case keywords
    edge_keywords = [
        'edge case', 'corner case', 'boundary', 
        'empty', 'zero', 'negative', 'invalid input',
        'single element', 'large input', 'null', 'none'
    ]
    
    # Count mentioned edge cases
    edge_mentions = sum(1 for keyword in edge_keywords if keyword in think_content)
    
    # Analyze test coverage
    test_coverage = 0.0
    if test_code:
        # Count test cases in test_code
        test_cases = test_code.count('assert candidate')
        unique_tests = len(set(re.findall(r'assert candidate\(([^)]+)', test_code)))
        
        # Estimate coverage quality
        test_coverage = min(unique_tests / 5, 1.0) * 0.5  # Max 0.5 for coverage
        
        # Check for edge cases in tests
        edge_tests = sum(1 for keyword in edge_keywords if keyword in test_code.lower())
        test_coverage += min(edge_tests / 3, 0.5)  # Max 0.5 for edge cases
    
    # Calculate final reward
    reward = min(0.3 * min(edge_mentions, 3) + test_coverage, 1.0)
    
    if kwargs.get('debug', False):
        print(f"Coverage Reward: {reward:.2f}")
        print(f"Edge mentions: {edge_mentions}, Test coverage: {test_coverage:.2f}")
    
    return reward


# Registry of all available reward functions
REWARD_FUNCTIONS = {
    'format': python_syntax_reward,
    'accuracy': python_execution_reward,
    # 'completeness': code_completeness_reward,
    # 'coverage': code_coverage_reward,
}


def calculate_rewards(generated_code: str, expected_code: str, 
                     reward_types: List[str] = None, 
                     problem: str = "",
                     test_code: str = None,
                     debug: bool = False) -> Dict[str, float]:
    """Calculate rewards with test_code support."""
    if reward_types is None:
        reward_types = ['format', 'accuracy', 'test', 'completeness', 'format']
    
    rewards = {}
    for reward_type in reward_types:
        if reward_type in REWARD_FUNCTIONS:
            reward_func = REWARD_FUNCTIONS[reward_type]
            try:
                rewards[reward_type] = reward_func(
                    generated_code, 
                    expected_code, 
                    problem=problem,
                    test_code=test_code,
                    debug=debug
                )
            except Exception as e:
                if debug:
                    print(f"Error calculating {reward_type} reward: {str(e)}")
                    traceback.print_exc()
                rewards[reward_type] = 0.0
        else:
            print(f"Warning: Unknown reward type '{reward_type}'")
    
    return rewards