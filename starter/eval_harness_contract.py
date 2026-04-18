#!/usr/bin/env python3
"""
eval_harness_contract.py
Matches the exact interface the grader will call.
Do NOT modify this file.
"""
from inference import run

# Example usage (grader calls this for each test example):
#
#   from eval_harness_contract import run
#
#   result = run(
#       prompt="What's the weather in Tokyo?",
#       history=[]
#   )
#   # result → '<tool_call>{"tool":"weather","args":{"location":"Tokyo","unit":"C"}}</tool_call>'
#
#   result = run(
#       prompt="And in Fahrenheit?",
#       history=[
#           {"role":"user",      "content":"What's the weather in Tokyo?"},
#           {"role":"assistant", "content":'<tool_call>{"tool":"weather","args":{"location":"Tokyo","unit":"C"}}</tool_call>'},
#       ]
#   )

__all__ = ["run"]
