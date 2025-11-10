"""
Utility functions for the Kimi Writing Agent.
"""

import json
from typing import Any, Callable, Dict, List

import httpx
import tiktoken


def _estimate_tokens_with_tiktoken(model: str, messages: List[Dict]) -> int:
    """
    Estimate token count using tiktoken library as a fallback.
    This is a simplified version based on the OpenAI cookbook.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(
            f"Warning: Model '{model}' not found for tiktoken. Using cl100k_base encoding."
        )
        encoding = tiktoken.get_encoding("cl100k_base")

    num_tokens = 0
    for message in messages:
        num_tokens += 4  # A rough estimate for message overhead
        for key, value in message.items():
            if not value:
                continue

            content_to_encode = ""
            if key == "tool_calls":
                try:
                    # For tool calls, encode the function name and arguments
                    for tool_call in value:
                        content_to_encode += tool_call.get("function", {}).get(
                            "name", ""
                        )
                        content_to_encode += tool_call.get("function", {}).get(
                            "arguments", ""
                        )
                except Exception:
                    content_to_encode += str(value)  # Fallback
            else:
                content_to_encode = str(value)

            num_tokens += len(encoding.encode(content_to_encode))

            if key == "name":
                num_tokens += 1  # Additional token for the name

    num_tokens += 3  # A rough estimate for priming the reply
    return num_tokens


def estimate_token_count(
    base_url: str, api_key: str, model: str, messages: List[Dict]
) -> int:
    """
    Estimate the token count for the given messages using the Moonshot API.

    Note: Token estimation uses api.moonshot.ai (not .cn)

    Args:
        base_url: The base URL for the API (will be converted to .ai for token endpoint)
        api_key: The API key for authentication
        model: The model name
        messages: List of message dictionaries

    Returns:
        Total token count
    """
    # Convert messages to serializable format (remove non-serializable objects)
    serializable_messages = []
    for msg in messages:
        if hasattr(msg, "model_dump"):
            # OpenAI SDK message object
            msg_dict = msg.model_dump()
        elif isinstance(msg, dict):
            msg_dict = msg.copy()
        else:
            msg_dict = {"role": "assistant", "content": str(msg)}

        # Clean up the message to only include serializable fields
        clean_msg = {}
        if "role" in msg_dict:
            clean_msg["role"] = msg_dict["role"]
        if "content" in msg_dict and msg_dict["content"]:
            clean_msg["content"] = msg_dict["content"]
        if "name" in msg_dict:
            clean_msg["name"] = msg_dict["name"]
        if "tool_calls" in msg_dict and msg_dict["tool_calls"]:
            clean_msg["tool_calls"] = msg_dict["tool_calls"]
        if "tool_call_id" in msg_dict:
            clean_msg["tool_call_id"] = msg_dict["tool_call_id"]

        serializable_messages.append(clean_msg)

    try:
        # Both token estimation and chat use api.moonshot.ai
        token_base_url = base_url

        # Make the API call
        with httpx.Client(
            base_url=token_base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0,
        ) as client:
            response = client.post(
                "/tokenizers/estimate-token-count",
                json={"model": model, "messages": serializable_messages},
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", {}).get("total_tokens", 0)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 405 or "openrouter.ai" in base_url:
            print(
                f"Warning: Token estimation API endpoint not available (or not applicable for {base_url}). Falling back to local tiktoken estimation."
            )
            return _estimate_tokens_with_tiktoken(model, serializable_messages)
        else:
            raise e
    except httpx.RequestError:
        print(
            f"Warning: Could not connect to token estimation endpoint. Falling back to local tiktoken estimation."
        )
        return _estimate_tokens_with_tiktoken(model, serializable_messages)


def get_tool_definitions() -> List[Dict[str, Any]]:
    """
    Returns the tool definitions in the format expected by kimi-k2-thinking.

    Returns:
        List of tool definition dictionaries
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Performs a web search using a search engine to gather information, articles, and data on a specified topic. Use this to conduct research before writing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"}
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_project",
                "description": "Creates a new project folder in the 'output' directory with a sanitized name. This should be called first before writing any files. Only one project can be active at a time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_name": {
                            "type": "string",
                            "description": "The name for the project folder (will be sanitized for filesystem compatibility)",
                        }
                    },
                    "required": ["project_name"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Writes content to a markdown file in the active project folder. Supports three modes: 'create' (creates new file, fails if exists), 'append' (adds content to end of existing file), 'overwrite' (replaces entire file content).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "The name of the markdown file to write (should end in .md)",
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file",
                        },
                        "mode": {
                            "type": "string",
                            "enum": ["create", "append", "overwrite"],
                            "description": "The write mode: 'create' for new files, 'append' to add to existing, 'overwrite' to replace",
                        },
                    },
                    "required": ["filename", "content", "mode"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compress_context",
                "description": "INTERNAL TOOL - This is automatically called by the system when token limit is approached. You should not call this manually. It compresses the conversation history to save tokens.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]


def get_tool_map() -> Dict[str, Callable]:
    """
    Returns a mapping of tool names to their implementation functions.

    Returns:
        Dictionary mapping tool name strings to callable functions
    """
    from tools import (
        compress_context_impl,
        create_project_impl,
        web_search_impl,
        write_file_impl,
    )

    return {
        "create_project": create_project_impl,
        "write_file": write_file_impl,
        "compress_context": compress_context_impl,
        "web_search": web_search_impl,
    }


def get_system_prompt() -> str:
    """
    Returns the system prompt for the writing agent.

    Returns:
        System prompt string
    """
    return """# Core Identity: Kimi, Your Expert Writing Partner

You are Kimi, a sophisticated creative and technical writing assistant from Moonshot AI. You are an expert in both fiction and non-fiction, capable of producing substantial, high-quality, and complete works. You excel at long-form content and can leverage a large context window to maintain consistency and coherence.

# Core Capabilities

*   **Project Management:** Create project folders to organize your work (`create_project`).
*   **File I/O:** Write, append, or overwrite markdown files (`write_file`).
*   **Research:** Perform web searches to gather information and data for your writing (`web_search`).
*   **Context Management:** Automatically compress conversation history to stay within token limits (`compress_context`).

# General Best Practices

*   **Research extensively:** As part of your creation process, either for outlines or actual publishable content, fiction or non-fiction, make sure you use the `web_search` tool as a foundation to your work. This is critical for non-fiction, and also a strong recommendation for fiction, in order to provide factual substance to the stories where needed.
*   **Be Specific:** The more detailed your request, the better the result. Provide context, desired tone, style, and length.
*   **Iterate:** Break down large projects into smaller, manageable parts (e.g., chapters, sections).
*   **Provide Context:** For ongoing projects, provide the previous sections of your work to ensure consistency.
*   **Use Descriptive Filenames:** Use clear and descriptive filenames (e.g., `chapter_01_the_discovery.md`, `report_on_quantum_computing.md`).

# Fiction Writing Guide

As a master storyteller, you should:

*   **Show, Don't Tell:** Instead of stating emotions, describe the character's actions, dialogue, and internal thoughts to convey them.
*   **Develop Compelling Characters:** Create three-dimensional characters with clear motivations, flaws, and arcs.
*   **Build Immersive Worlds:** Use vivid descriptions and sensory details to bring your settings to life.
*   **Craft Engaging Plots:** Weave together plot points, subplots, and character arcs to create a compelling narrative.
*   **Write Natural Dialogue:** Ensure dialogue is realistic, reveals character, and advances the plot.
*   **Maintain Consistency:** Keep track of character details, plot points, and world-building elements across the entire work.

*Example Fiction Workflow:*
1.  **`create_project(project_name='My Sci-Fi Novel')`**
2.  **`web_search(query='effects of zero gravity in the human body')`**
3.  **`write_file(filename='chapter_01.md', content='...full text of chapter 1...', mode='create')`**
4.  **`write_file(filename='chapter_02.md', content='...full text of chapter 2...', mode='create')`**

# Non-Fiction Writing Guide

As an expert technical writer and researcher, you should:

*   **Prioritize Factual Accuracy:** Use the `web_search` tool to research topics and verify information. Clearly cite your sources when appropriate.
*   **Structure Your Work Logically:** Use clear headings, subheadings, and a logical flow to organize your writing.
*   **Write with Clarity and Precision:** Use clear, concise language. Avoid jargon when writing for a general audience.
*   **Synthesize Information:** When given multiple sources, synthesize the information into a coherent and comprehensive narrative.
*   **Think Step-by-Step:** For complex topics, break down your reasoning process. For example: "First, I will research X. Second, I will analyze Y. Finally, I will write a summary."

*Example Non-Fiction Workflow:*
1.  **`create_project(project_name='History of AI')`**
2.  **`web_search(query='history of artificial intelligence')`**
3.  **`write_file(filename='01_early_concepts.md', content='...detailed history of early AI concepts...', mode='create')`**
4.  **`write_file(filename='02_the_rise_of_deep_learning.md', content='...detailed history of deep learning...', mode='create')`**

# Final Reminder

You have a 64K token context window per response. Use it to its full potential. Write rich, detailed, and complete content. A good short story is 5,000-10,000 words. A good chapter is 3,000-5,000 words. A good report is comprehensive and well-researched. Write what the work needs to be excellent."""
