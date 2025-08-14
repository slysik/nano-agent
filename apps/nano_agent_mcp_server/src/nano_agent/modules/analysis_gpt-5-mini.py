"""
Analysis of constants defined in apps/nano_agent_mcp_server/src/nano_agent/modules/constants.py

Summary:
- Total constants found: 26
- Categories identified (12): defaults, available models, model info, provider requirements,
  agent configuration, tool names, available tools list, demo configuration, system prompts,
  error messages, success messages, version info
- Default model: "gpt-5-mini"
- Default provider: "openai"

This module provides a single function get_constants_report() that returns a dictionary
containing the analysis results.
"""

from typing import Dict, List


def get_constants_report() -> Dict[str, object]:
    """Return a report summarizing the constants analysis.

    Returns a dictionary with keys:
      - total_constants: int
      - categories_count: int
      - categories: list
      - default_model: str
      - default_provider: str
    """
    report = {
        "total_constants": 26,
        "categories_count": 12,
        "categories": [
            "default_model_configuration",
            "available_models",
            "model_info",
            "provider_requirements",
            "agent_configuration",
            "tool_names",
            "available_tools_list",
            "demo_configuration",
            "system_prompts",
            "error_messages",
            "success_messages",
            "version_info",
        ],
        "default_model": "gpt-5-mini",
        "default_provider": "openai",
    }
    return report


# Analysis completed by gpt-5-mini