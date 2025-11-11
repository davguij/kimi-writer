# Kimi Editor Agent Guide

## Overview

The Kimi Editor Agent is a world-class book editor powered by the kimi-k2-thinking model. It specializes in editing and improving both fiction and non-fiction manuscripts with the goal of creating Amazon KDP best-sellers.

## Key Features

- **File Reading & Editing**: Can read existing manuscript files and edit them in place
- **Audience-Centric Editing**: Researches target audiences to ensure content resonates
- **Commercial Focus**: Optimized for Amazon KDP success and series potential
- **Full Creative Control**: Not constrained by preserving original voice if changes improve marketability
- **Comprehensive Editorial Services**: Developmental editing, line editing, copy editing, and formatting

## Usage

### Basic Usage

```bash
# Interactive mode
python kimi-editor.py

# With inline prompt
python kimi-editor.py "Edit my manuscript in the project folder 'my_novel'"

# Recovery mode
python kimi-editor.py --recover output/my_project/.context_summary_20250107_143022.md
```

### Typical Workflow

1. **Place your manuscript files** in a folder under `output/` (e.g., `output/my_novel/`)
2. **Run the editor** and specify you want to edit the existing project
3. **The agent will**:
   - Use `create_project` to set the existing folder as active
   - Use `list_files()` to see what files exist
   - Use `read_file()` to read each file before editing
   - Use `write_file(mode='overwrite')` to edit files in place
   - Create editorial notes and reports as new files

## Available Tools

### File Management Tools

- **`list_files()`**: Lists all files in the active project folder
- **`read_file(filename)`**: Reads content from a file
- **`write_file(filename, content, mode)`**: Writes content to a file
  - `mode='create'`: Create new file (for editorial notes)
  - `mode='overwrite'`: Replace file (for editing manuscripts)
  - `mode='append'`: Add to end of file

### Project Management

- **`create_project(project_name)`**: Creates new project folder or sets existing one as active

### Research Tools

- **`web_search(query)`**: Searches the web for:
  - Target audience research
  - Genre conventions and market analysis
  - Fact checking
  - Grammar and style guidelines
  - Competitive title research

### System Tools

- **`compress_context()`**: Automatically manages token limits (internal use)

## Editorial Philosophy

### Core Identity: Lukas

The agent operates as "Lukas," a world-class editor with 50+ best-selling titles. Key principles:

- **Commercial Success First**: Every decision aims for best-seller status
- **Full Creative Freedom**: Will change voice, style, or approach if needed for success
- **Audience-Centric**: Deep understanding of target readers is paramount
- **Series Thinking**: Considers franchise potential in all editorial decisions
- **Amazon KDP Optimized**: Formatting and structure ready for KDP/EPUB conversion

### Editorial Standards

#### For Fiction
- Compelling openings that hook immediately
- Deep character development readers care about
- Engaging pacing with emotional impact
- Immersive world-building
- Plot coherence with no holes
- Series potential consideration

#### For Non-Fiction
- Value-packed, actionable content
- Thoroughly researched and fact-checked
- Clear, logical structure
- Engaging writing style
- Practical application focus
- Authority building
- Series potential consideration

#### Professional Polish
- Impeccable grammar and style
- Complete consistency throughout
- Compelling, SEO-friendly titles
- Smooth reading experience
- Proper formatting for Pandoc/EPUB conversion

## Critical Editing Rules

1. **ALWAYS use `list_files()` first** to see what exists
2. **ALWAYS use `read_file()` before editing** to see current content
3. **ALWAYS use `mode='overwrite'` for manuscript files** to edit in place
4. **NEVER create new "_edited" files** - overwrite originals
5. **Only use `mode='create'` for new editorial notes and reports**

## Example Editing Session

```
User: "Edit the manuscript in my_novel project folder"

Agent Workflow:
1. create_project(project_name='my_novel')
   → Sets existing folder as active
   
2. list_files()
   → Sees: chapter_01.md, chapter_02.md, chapter_03.md
   
3. web_search(query='target audience for thriller novels')
   → Researches market
   
4. read_file(filename='chapter_01.md')
   → Reads current content
   
5. [Analyzes and improves content]
   
6. write_file(filename='chapter_01.md', content='[edited version]', mode='overwrite')
   → Overwrites with improved version
   
7. [Repeats for each chapter]
   
8. write_file(filename='editorial_summary.md', content='[summary]', mode='create')
   → Creates new editorial report
```

## Output Structure

After editing, your project folder will contain:

```
output/my_novel/
├── chapter_01.md          # Edited in place
├── chapter_02.md          # Edited in place
├── chapter_03.md          # Edited in place
├── editorial_summary.md   # New: Overview of changes
├── market_positioning.md  # New: KDP recommendations
└── series_potential.md    # New: Ideas for sequels
```

## Publishing Preparation

The editor prepares manuscripts for:

- **Pandoc conversion**: Markdown → EPUB
- **Amazon KDP**: Title optimization, category fit, review-worthy quality
- **Series development**: Structured for sequels and franchise building

## Success Criteria

The editor is successful when the book:

1. Becomes a best-seller
2. Generates strong reviews (5-star quality)
3. Establishes author credibility
4. Creates opportunities for series expansion
5. Delivers financial success

## Tips for Best Results

- **Provide Context**: Specify genre, target audience, and goals
- **Share Market Info**: Mention competitive titles or positioning
- **Be Specific**: Detail what aspects need most attention
- **Allow Freedom**: Trust the editor's commercial instincts
- **Think Series**: Consider long-term franchise potential

## Recovery & Continuation

If editing is interrupted:

```bash
# Context is automatically saved every 10 iterations
python kimi-editor.py --recover output/my_project/.context_summary_TIMESTAMP.md
```

## Configuration

Set environment variables in `.env`:

```bash
MOONSHOT_API_KEY=your-api-key-here
MOONSHOT_BASE_URL=https://api.moonshot.ai/v1
MOONSHOT_MODEL_NAME=kimi-k2-thinking
```

## Differences from Writer Agent

| Feature | Writer Agent | Editor Agent |
|---------|-------------|--------------|
| Primary Function | Create new content | Edit existing content |
| File Operations | Create new files | Read & overwrite existing files |
| Tools | No read/list tools | Includes read_file, list_files |
| Voice | Preserves creative freedom | Changes for commercial success |
| Focus | Content creation | Market optimization |

## Technical Details

- **Model**: kimi-k2-thinking (64K token output)
- **Context Limit**: 200,000 tokens
- **Auto-compression**: At 180,000 tokens
- **Max Iterations**: 300
- **Backup Frequency**: Every 10 iterations