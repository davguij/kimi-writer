#!/usr/bin/env python3
"""
Kimi Editor Agent - An autonomous agent for book editing tasks.

This agent uses the kimi-k2-thinking model to edit, revise, and improve
novels, books, manuscripts, and other written content.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

from tools.compression import compress_context_impl
from utils import (
    estimate_token_count,
)

# Constants
MAX_ITERATIONS = 300
TOKEN_LIMIT = 200000
COMPRESSION_THRESHOLD = 180000  # Trigger compression at 90% of limit
MODEL_NAME = os.getenv("MOONSHOT_MODEL_NAME", "kimi-k2-thinking")
BACKUP_INTERVAL = 10  # Save backup summary every N iterations


def load_context_from_file(file_path: str) -> str:
    """
    Loads context from a summary file for recovery.

    Args:
        file_path: Path to the context summary file

    Returns:
        Content of the file as string
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        print(f"✓ Loaded context from: {file_path}\n")
        return content
    except Exception as e:
        print(f"✗ Error loading context file: {e}")
        sys.exit(1)


def get_user_input() -> tuple[str, bool]:
    """
    Gets user input from command line, either as a prompt or recovery file.

    Returns:
        Tuple of (prompt/context, is_recovery_mode)
    """
    parser = argparse.ArgumentParser(
        description="Kimi Editor Agent - Edit, revise, and improve written content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fresh start with inline prompt
  python kimi-editor.py "Edit my manuscript for grammar and style"

  # Recovery mode from previous context
  python kimi-editor.py --recover my_project/.context_summary_20250107_143022.md
        """,
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help='Your editing request (e.g., "Improve the pacing of my novel")',
    )
    parser.add_argument(
        "--recover", type=str, help="Path to a context summary file to continue from"
    )

    args = parser.parse_args()

    # Check if recovery mode
    if args.recover:
        context = load_context_from_file(args.recover)
        return context, True

    # Check if prompt provided as argument
    if args.prompt:
        return args.prompt, False

    # Interactive prompt
    print("=" * 60)
    print("Kimi Editor Agent")
    print("=" * 60)
    print("\nEnter your editing request (or 'quit' to exit):")
    print("Example: Review and edit my novel for consistency and pacing\n")

    prompt = input("> ").strip()

    if prompt.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        sys.exit(0)

    if not prompt:
        print("Error: Empty prompt. Please provide an editing request.")
        sys.exit(1)

    return prompt, False


def convert_message_for_api(msg: Any) -> Dict[str, Any]:
    """
    Converts a message object to a dictionary suitable for API calls.
    Preserves reasoning_content if present.

    Args:
        msg: Message object (can be OpenAI message object or dict)

    Returns:
        Dictionary representation of the message
    """
    if isinstance(msg, dict):
        return msg

    # Convert OpenAI message object to dict
    msg_dict = {
        "role": msg.role,
    }

    if msg.content:
        msg_dict["content"] = msg.content

    # Preserve reasoning_content if present
    if hasattr(msg, "reasoning_content"):
        reasoning = getattr(msg, "reasoning_content")
        if reasoning:
            msg_dict["reasoning_content"] = reasoning

    # Preserve tool calls if present
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        msg_dict["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]

    # Preserve tool call id for tool response messages
    if hasattr(msg, "tool_call_id") and msg.tool_call_id:
        msg_dict["tool_call_id"] = msg.tool_call_id

    if hasattr(msg, "name") and msg.name:
        msg_dict["name"] = msg.name

    return msg_dict


def get_editor_tool_definitions() -> List[Dict[str, Any]]:
    """
    Returns the tool definitions for the editor agent.

    Returns:
        List of tool definition dictionaries
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "Lists all files in the active project folder. Use this to see what files are available to edit.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Reads the content of a file from the active project folder. ALWAYS read a file before editing it to see its current content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "The name of the file to read",
                        }
                    },
                    "required": ["filename"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Performs a web search to verify facts, check grammar rules, research style guidelines, or gather reference information for editing decisions.",
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
                "description": "Creates a new project folder in the 'output' directory for organizing edited content. Can also be used to set an existing project folder as active.",
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
                "description": "Writes edited content to a markdown file in the active project folder. For editing existing manuscript files, ALWAYS use 'overwrite' mode to edit them in place. Use 'create' only for new editorial notes and reports. Modes: 'create' (creates new file, fails if exists), 'append' (adds content to end), 'overwrite' (replaces entire file content - USE THIS for editing manuscripts).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "description": "The name of the markdown file to write (should end in .md)",
                        },
                        "content": {
                            "type": "string",
                            "description": "The edited content to write to the file",
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


def get_editor_system_prompt() -> str:
    """
    Returns the system prompt for the editor agent.

    Returns:
        System prompt string
    """
    return """# Core Identity: Lukas, Your World-class Book Editor

You are a world-class book editor specialized in both fiction and non-fiction, and you have more than 50 best-selling titles under your belt.

Your job is to review the written content, in whatever form, including its title, to make sure that:
- It's attractive and well-structured
- The research is accurate and well-sourced
- It's coherent, professionally written and edited
- It gets the reader engaged and wanting more
- It provides plenty of value to the reader, by being actionable when appropriate
- In general, it checks all the boxes when it comes to a greatly successful book, adapting to the particularities of fiction or non-fiction, depending on the case

You also need to make sure that you **fully define and understand the target audience**, using the `web_search` tool for researching purposes, so that the resulting book is a great fit for that audience and speaks to their needs, to their hearts, to their souls.

# Publishing Goals & Success Criteria

We will be selling the book through **Amazon KDP**, and we're aiming at hitting a home run. You will be successful if:
- The resulting book becomes a best-seller and a financial success
- We're able to evolve it into an ongoing series of books
- We establish the author and the publishing house as a reference for the future

# Formatting Requirements

You need to refine the formatting of the book. We will later be using the **Pandoc CLI** to convert from Markdown to EPUB. The formatting needs to be:
- As clear and attractive as possible
- Observing the limitations of the KDP/EPUB format and the conversion step
- Properly structured with appropriate headings, paragraphs, and styling

**You have full creative freedom and you're free to edit the content as you see fit for the purpose.**

# Core Capabilities

*   **Project Management:** Create or set project folders to organize your edited work (`create_project`).
*   **File Discovery:** List all files in the project folder to see what's available (`list_files`).
*   **File Reading:** Read existing manuscript files to understand their content before editing (`read_file`).
*   **File Writing:** Write edited content, editorial notes, or revision reports to markdown files (`write_file`).
*   **Research:** Perform web searches to verify facts, check grammar rules, research style guidelines, understand target audiences, or gather reference information (`web_search`).
*   **Context Management:** Automatically compress conversation history to stay within token limits (`compress_context`).

# Editorial Excellence Standards

## Audience-Centric Editing

Before making any editorial decisions:
1. **Define the Target Audience:** Who are they? What are their pain points? What do they desire?
2. **Research the Market:** Use `web_search` to understand genre conventions, reader expectations, and competitive titles
3. **Speak to Their Hearts:** Ensure every element resonates with the target audience's needs and emotions

## For Fiction

*   **Compelling Opening:** Hook readers immediately - they should be unable to put it down
*   **Character Depth:** Create characters readers care about deeply, with clear motivations and authentic arcs
*   **Engaging Pacing:** Balance action, dialogue, and description to maintain momentum
*   **Emotional Impact:** Show, don't tell - make readers feel every emotion
*   **World-building:** Create immersive settings that feel real and lived-in
*   **Plot Coherence:** Ensure no plot holes, maintain consistency, deliver satisfying payoffs
*   **Series Potential:** Consider how this book can lead to sequels and establish a franchise

## For Non-Fiction

*   **Value-Packed Content:** Every chapter should provide actionable insights and takeaways
*   **Factual Accuracy:** Research thoroughly and verify all claims using `web_search`
*   **Clear Structure:** Logical flow that builds knowledge progressively
*   **Engaging Writing:** Even educational content should be compelling and readable
*   **Practical Application:** Show readers how to implement the ideas
*   **Authority Building:** Establish credibility through research, examples, and expertise
*   **Series Potential:** Structure content to naturally extend into follow-up books

## Professional Polish

*   **Grammar & Style:** Impeccable grammar, punctuation, and professional writing standards
*   **Consistency:** Characters, facts, timelines, terminology - everything must be consistent
*   **Voice & Style:** Apply the most effective voice and style for the target audience and genre - change as needed for commercial success
*   **Title & Subtitle:** Ensure they're compelling, SEO-friendly, and market-appropriate for Amazon KDP
*   **Chapter Titles:** Make them enticing and indicative of value
*   **Reading Experience:** Smooth flow, appropriate pacing, no jarring transitions

# Amazon KDP Best Practices

*   **Title Optimization:** Make it searchable and compelling for Amazon's algorithm
*   **Description-Ready Content:** Ensure the book delivers on what the description promises
*   **Category Fit:** Content should clearly fit its target category/genre
*   **Review-Worthy Quality:** Aim for content that naturally generates 5-star reviews
*   **Look Inside Preview:** The first pages must hook potential buyers immediately
*   **Series Positioning:** If part of a series, ensure proper setup for continuation

# Editorial Workflow

## Initial Assessment
1. **Set the active project:** Use `create_project` with the existing project name to make it active
2. **`list_files()`** - See what files exist in the project
3. **`web_search(query='target audience for [genre/topic]')`** - Understand the market
4. **`web_search(query='best-selling books in [category]')`** - Learn from success stories
5. **Read files:** Use `read_file` to read each manuscript file and understand the content
6. **Define Strategy:** Determine what changes will maximize commercial and artistic success

## Developmental Editing
1. **Big Picture Assessment:** Structure, pacing, character/concept development
2. **`write_file(filename='editorial_assessment.md', content='...comprehensive analysis...', mode='create')`**
3. **Target Audience Alignment:** Ensure content speaks to the right readers
4. **Series Potential:** Identify opportunities for expansion

## Content Editing (CRITICAL: Read Before Editing!)
1. **`list_files()`** - Identify all manuscript files
2. **For each file:**
   - **`read_file(filename='chapter_01.md')`** - Read the current content first
   - **Edit and improve** the content based on editorial standards
   - **`write_file(filename='chapter_01.md', content='...fully edited chapter...', mode='overwrite')`** - Overwrite with edited version
3. **NEVER create new "_edited" files** - Always overwrite the original manuscript files directly
4. **Value Optimization:** Ensure every page provides value and engagement
5. **Research Verification:** Use `web_search` to fact-check and enhance content
6. **Formatting:** Apply proper Markdown formatting for Pandoc conversion

## Line Editing & Copy Editing
1. **Sentence-Level Polish:** Enhance clarity, flow, and impact
2. **Grammar & Mechanics:** Perfect professional standards
3. **Consistency Check:** Verify all details align throughout
4. **Final Format Check:** Ensure Markdown is clean and conversion-ready

## Final Deliverables
1. **`write_file(filename='editorial_summary.md', content='...overview of all changes...', mode='create')`**
2. **`write_file(filename='market_positioning.md', content='...recommendations for Amazon KDP...', mode='create')`**
3. **`write_file(filename='series_potential.md', content='...ideas for follow-up books...', mode='create')`**

**CRITICAL EDITING RULES:**
1. **ALWAYS use `list_files()` first** to see what files exist in the project
2. **ALWAYS use `read_file(filename='...')` before editing** to see the current content
3. **ALWAYS use `mode='overwrite'` when editing existing manuscript files** to edit them in place
4. **NEVER create new files with "_edited" or similar suffixes** - overwrite the originals
5. **Only use `mode='create'` for brand new editorial notes, reports, and supplementary documents**

# Quality Benchmarks

Ask yourself for every section:
- Would a reader recommend this to a friend?
- Does this content deliver on the book's promise?
- Is this engaging enough to compete with best-sellers in this category?
- Does this build the author's authority and brand?
- Is this formatted properly for KDP/EPUB conversion?

# Final Reminder

You have a 64K token context window per response. Use it to produce comprehensive, publication-ready editorial work. Your edits should transform good manuscripts into best-selling books. Think like a publisher investing in a franchise, not just editing a single book. Your success is measured by commercial viability, reader engagement, and series potential.

**Remember: You have full creative control.** If the author's original voice, style, or approach isn't working for the target audience or market, change it. Your job is to create a best-seller, not to preserve every original choice. Make bold editorial decisions that serve the book's commercial and artistic success."""


def main():
    """Main agent loop."""

    # Get API key
    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        print("Error: MOONSHOT_API_KEY environment variable not set.")
        print("Please set your API key: export MOONSHOT_API_KEY='your-key-here'")
        sys.exit(1)

    base_url = os.getenv("MOONSHOT_BASE_URL", "https://api.moonshot.ai/v1")

    # Debug: Show that key is loaded (masked for security)
    if len(api_key) > 8:
        print(f"✓ API Key loaded: {api_key[:4]}...{api_key[-4:]}")
    else:
        print(f"⚠️  Warning: API key seems too short ({len(api_key)} chars)")
    print(f"✓ Base URL: {base_url}\n")

    # Initialize OpenAI client
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    # Get user input
    user_prompt, is_recovery = get_user_input()

    # Initialize message history
    messages = [{"role": "system", "content": get_editor_system_prompt()}]

    if is_recovery:
        messages.append(
            {
                "role": "user",
                "content": f"[RECOVERED CONTEXT]\n\n{user_prompt}\n\n[END RECOVERED CONTEXT]\n\nPlease continue the editing work from where we left off.",
            }
        )
        print("🔄 Recovery mode: Continuing from previous context\n")
    else:
        messages.append({"role": "user", "content": user_prompt})
        print(f"\n📝 Editing Task: {user_prompt}\n")

    # Get tool definitions and create tool mapping
    tools = get_editor_tool_definitions()

    # Import tool implementations
    from tools import (
        compress_context_impl,
        create_project_impl,
        list_files_impl,
        read_file_impl,
        web_search_impl,
        write_file_impl,
    )

    # Create tool map for the editor
    tool_map = {
        "create_project": create_project_impl,
        "write_file": write_file_impl,
        "read_file": read_file_impl,
        "list_files": list_files_impl,
        "compress_context": compress_context_impl,
        "web_search": web_search_impl,
    }

    print("=" * 60)
    print("Starting Kimi Editor Agent")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Max iterations: {MAX_ITERATIONS}")
    print(f"Context limit: {TOKEN_LIMIT:,} tokens")
    print(f"Auto-compression at: {COMPRESSION_THRESHOLD:,} tokens")
    print("=" * 60 + "\n")

    # Main agent loop
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n{'─' * 60}")
        print(f"Iteration {iteration}/{MAX_ITERATIONS}")
        print(f"{'─' * 60}")

        # Check token count before making API call
        try:
            token_count = estimate_token_count(base_url, api_key, MODEL_NAME, messages)
            print(
                f"📊 Current tokens: {token_count:,}/{TOKEN_LIMIT:,} ({token_count / TOKEN_LIMIT * 100:.1f}%)"
            )

            # Trigger compression if approaching limit
            if token_count >= COMPRESSION_THRESHOLD:
                print(f"\n⚠️  Approaching token limit! Compressing context...")
                compression_result = compress_context_impl(
                    messages=messages, client=client, model=MODEL_NAME, keep_recent=10
                )

                if "compressed_messages" in compression_result:
                    messages = compression_result["compressed_messages"]
                    print(f"✓ {compression_result['message']}")
                    print(
                        f"✓ Estimated tokens saved: ~{compression_result.get('tokens_saved', 0):,}"
                    )

                    # Recalculate token count
                    token_count = estimate_token_count(
                        base_url, api_key, MODEL_NAME, messages
                    )
                    print(f"📊 New token count: {token_count:,}/{TOKEN_LIMIT:,}\n")

        except Exception as e:
            print(f"⚠️  Warning: Could not estimate token count: {e}")
            token_count = 0

        # Auto-backup every N iterations
        if iteration % BACKUP_INTERVAL == 0:
            print(f"💾 Auto-backup (iteration {iteration})...")
            try:
                compression_result = compress_context_impl(
                    messages=messages,
                    client=client,
                    model=MODEL_NAME,
                    keep_recent=len(messages),  # Keep all messages, just save summary
                )
                if compression_result.get("summary_file"):
                    print(
                        f"✓ Backup saved: {os.path.basename(compression_result['summary_file'])}\n"
                    )
            except Exception as e:
                print(f"⚠️  Warning: Backup failed: {e}\n")

        # Call the model
        try:
            print("🤖 Calling kimi-k2-thinking model...\n")

            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=65536,  # 64K tokens
                tools=tools,
                temperature=1.0,
                stream=True,  # Enable streaming
            )

            # Accumulate the streaming response
            reasoning_content = ""
            content_text = ""
            tool_calls_data = []
            role = None
            finish_reason = None

            # Track if we've printed headers
            reasoning_header_printed = False
            content_header_printed = False
            tool_call_header_printed = False
            last_tool_index = -1

            # Process the stream
            for chunk in stream:
                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                finish_reason = chunk.choices[0].finish_reason

                # Get role if present (first chunk)
                if hasattr(delta, "role") and delta.role:
                    role = delta.role

                # Handle reasoning_content streaming
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    if not reasoning_header_printed:
                        print("=" * 60)
                        print(f"🧠 Reasoning (Iteration {iteration})")
                        print("=" * 60)
                        reasoning_header_printed = True

                    print(delta.reasoning_content, end="", flush=True)
                    reasoning_content += delta.reasoning_content

                # Handle regular content streaming
                if hasattr(delta, "content") and delta.content:
                    # Close reasoning section if it was open
                    if reasoning_header_printed and not content_header_printed:
                        print("\n" + "=" * 60 + "\n")

                    if not content_header_printed:
                        print("💬 Response:")
                        print("-" * 60)
                        content_header_printed = True

                    print(delta.content, end="", flush=True)
                    content_text += delta.content

                # Handle tool_calls
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        # Ensure we have enough slots in tool_calls_data
                        while len(tool_calls_data) <= tc_delta.index:
                            tool_calls_data.append(
                                {
                                    "id": None,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                    "chars_received": 0,
                                }
                            )

                        tc = tool_calls_data[tc_delta.index]

                        # Print header when we start receiving a tool call
                        if tc_delta.index != last_tool_index:
                            if reasoning_header_printed or content_header_printed:
                                print("\n" + "=" * 60 + "\n")

                            if hasattr(tc_delta, "function") and tc_delta.function.name:
                                print(
                                    f"🔧 Preparing tool call: {tc_delta.function.name}"
                                )
                                print("─" * 60)
                                tool_call_header_printed = True
                                last_tool_index = tc_delta.index

                        if tc_delta.id:
                            tc["id"] = tc_delta.id
                        if hasattr(tc_delta, "function"):
                            if tc_delta.function.name:
                                tc["function"]["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                tc["function"]["arguments"] += (
                                    tc_delta.function.arguments
                                )
                                tc["chars_received"] += len(tc_delta.function.arguments)

                                # Show progress indicator every 500 characters
                                if (
                                    tc["chars_received"] % 500 == 0
                                    or tc["chars_received"] < 100
                                ):
                                    # Calculate approximate words (rough estimate: 5 chars per word)
                                    words = tc["chars_received"] // 5
                                    print(
                                        f"\r💬 Generating arguments... {tc['chars_received']:,} characters (~{words:,} words)",
                                        end="",
                                        flush=True,
                                    )

            # Print closing for content if it was printed
            if content_header_printed:
                print("\n" + "-" * 60 + "\n")

            # Print completion for tool calls if any were received
            if tool_call_header_printed:
                print("\n✓ Tool call complete")
                print("─" * 60 + "\n")

            # Reconstruct the message object from accumulated data
            class ReconstructedMessage:
                def __init__(self):
                    self.role = role or "assistant"
                    self.content = content_text if content_text else None
                    self.reasoning_content = (
                        reasoning_content if reasoning_content else None
                    )
                    self.tool_calls = None

                    if tool_calls_data:
                        # Convert to proper format
                        from openai.types.chat import ChatCompletionMessageToolCall
                        from openai.types.chat.chat_completion_message_tool_call import (
                            Function,
                        )

                        self.tool_calls = []
                        for tc in tool_calls_data:
                            if tc["id"]:  # Only add if we have an ID
                                tool_call = type(
                                    "ToolCall",
                                    (),
                                    {
                                        "id": tc["id"],
                                        "type": "function",
                                        "function": type(
                                            "Function",
                                            (),
                                            {
                                                "name": tc["function"]["name"],
                                                "arguments": tc["function"][
                                                    "arguments"
                                                ],
                                            },
                                        )(),
                                    },
                                )()
                                self.tool_calls.append(tool_call)

            message = ReconstructedMessage()

            # Convert message to dict and add to history
            # Important: preserve the full message object structure
            messages.append(convert_message_for_api(message))

            # Check if the model called any tools
            if not message.tool_calls:
                print("=" * 60)
                print("✅ EDITING TASK COMPLETED")
                print("=" * 60)
                print(f"Completed in {iteration} iteration(s)")
                print("=" * 60)
                break

            # Handle tool calls
            print(f"\n🔧 Model decided to call {len(message.tool_calls)} tool(s):")

            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                args_str = tool_call.function.arguments

                try:
                    args = json.loads(args_str)
                except json.JSONDecodeError:
                    args = {}

                print(f"\n  → {func_name}")
                print(
                    f"    Arguments: {json.dumps(args, ensure_ascii=False, indent=6)}"
                )

                # Get the tool implementation
                tool_func = tool_map.get(func_name)

                if not tool_func:
                    result = f"Error: Unknown tool '{func_name}'"
                    print(f"    ✗ {result}")
                else:
                    # Special handling for compress_context (needs extra params)
                    if func_name == "compress_context":
                        result_data = compress_context_impl(
                            messages=messages,
                            client=client,
                            model=MODEL_NAME,
                            keep_recent=10,
                        )
                        result = result_data.get("message", "Compression completed")

                        # Update messages with compressed version
                        if "compressed_messages" in result_data:
                            messages = result_data["compressed_messages"]
                    else:
                        # Call the tool with its arguments
                        result = tool_func(**args)

                    # Print result (truncate if too long)
                    if len(str(result)) > 200:
                        print(f"    ✓ {str(result)[:200]}...")
                    else:
                        print(f"    ✓ {result}")

                # Add tool result to messages
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": str(result),
                }
                messages.append(tool_message)

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user. Saving context...")
            # Save current context before exiting
            try:
                compression_result = compress_context_impl(
                    messages=messages,
                    client=client,
                    model=MODEL_NAME,
                    keep_recent=len(messages),
                )
                if compression_result.get("summary_file"):
                    print(f"✓ Context saved to: {compression_result['summary_file']}")
                    print(f"\nTo resume, run:")
                    print(
                        f"  python kimi-editor.py --recover {compression_result['summary_file']}"
                    )
            except:
                pass
            sys.exit(0)

        except Exception as e:
            print(f"\n✗ Error during iteration {iteration}: {e}")
            print(f"Attempting to continue...\n")
            continue

    # If we hit max iterations
    if iteration >= MAX_ITERATIONS:
        print("\n" + "=" * 60)
        print("⚠️  MAX ITERATIONS REACHED")
        print("=" * 60)
        print(f"\nReached maximum of {MAX_ITERATIONS} iterations.")
        print("Saving final context...")

        try:
            compression_result = compress_context_impl(
                messages=messages,
                client=client,
                model=MODEL_NAME,
                keep_recent=len(messages),
            )
            if compression_result.get("summary_file"):
                print(f"✓ Context saved to: {compression_result['summary_file']}")
                print(f"\nTo resume, run:")
                print(
                    f"  python kimi-editor.py --recover {compression_result['summary_file']}"
                )
        except Exception as e:
            print(f"✗ Error saving context: {e}")


if __name__ == "__main__":
    main()
