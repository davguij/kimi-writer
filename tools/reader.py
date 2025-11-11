"""
File reading tool for reading existing files.
"""

import os

from .project import get_active_project_folder


def read_file_impl(filename: str) -> str:
    """
    Reads content from a file in the active project folder.

    Args:
        filename: The name of the file to read

    Returns:
        File content or error message
    """
    # Check if project folder is initialized
    project_folder = get_active_project_folder()
    if not project_folder:
        return "Error: No active project folder. Please create or set a project first using create_project."

    # Ensure filename ends with .md if it doesn't have an extension
    if "." not in filename:
        filename = filename + ".md"

    # Create full file path
    file_path = os.path.join(project_folder, filename)

    try:
        # Check if file exists
        if not os.path.exists(file_path):
            return (
                f"Error: File '{filename}' does not exist in the active project folder."
            )

        # Read the file
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        return content

    except Exception as e:
        return f"Error reading file '{filename}': {str(e)}"


def list_files_impl() -> str:
    """
    Lists all files in the active project folder.

    Returns:
        List of files or error message
    """
    # Check if project folder is initialized
    project_folder = get_active_project_folder()
    if not project_folder:
        return "Error: No active project folder. Please create or set a project first using create_project."

    try:
        # Check if folder exists
        if not os.path.exists(project_folder):
            return f"Error: Project folder does not exist at '{project_folder}'."

        # List all files (not directories)
        files = []
        for item in os.listdir(project_folder):
            item_path = os.path.join(project_folder, item)
            if os.path.isfile(item_path):
                # Get file size
                size = os.path.getsize(item_path)
                files.append(f"  - {item} ({size:,} bytes)")

        if not files:
            return f"Project folder '{os.path.basename(project_folder)}' is empty (no files)."

        file_list = "\n".join(files)
        return f"Files in project folder '{os.path.basename(project_folder)}':\n{file_list}"

    except Exception as e:
        return f"Error listing files: {str(e)}"
