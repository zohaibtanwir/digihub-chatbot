import re
from langchain.schema import HumanMessage, SystemMessage

def generate_final_response(query, markdown_response, detailed_response, summary_response, llm):
    """
    Generate a final response based on relevant chunks retrieved from the documents.
    """
    if is_generic_question(query):
        return {"error": "This question is outside the scope of the documents provided."}

    combined_input = f"""
    Task:
    You are given a set of text chunks retrieved for a question. Your task is to:
    - Filter out irrelevant chunks and provide an answer strictly based on the relevant chunks.
    - Preserve image paths exactly as they appear in the original chunks.
    - Extract and include the file name from the source metadata in the response.

    Input:
    Question: {query}
    Relevant Chunks with Possible Images: {markdown_response}
    Relevant Chunks Without Images: {detailed_response}
    Summary Chunks: {summary_response}

    Instructions:
    1. Use only relevant chunks to form the response.
    2. Retain image paths in the format `![Image](image_path)`.
    3. Extract file names from the metadata and include them in the response.
    4. Ensure the response is clear, concise, and strictly based on the documents.

    Response Format:
    - Response: [Final answer based on relevant chunks]
    - Source: [File names from metadata]
    """
    messages = [
        SystemMessage(content="You are an AI system designed to generate document-specific responses."),
        HumanMessage(content=combined_input)
    ]

    result = llm.invoke(messages)
    return result

def is_generic_question(query):
    """
    Check if the query is a generic question.

    Args:
        query (str): The user's query.

    Returns:
        bool: True if the query is generic, False otherwise.
    """
    generic_keywords = [
        "president", "capital", "weather", "news", "time", "date", "population",
        "currency", "language", "country", "city", "continent", "temperature",
        "holiday", "event", "festival", "leader", "government", "economy",
        "history", "geography", "sports", "score", "result", "update", "headline",
        "celebrity", "movie", "music", "general knowledge", "random fact"
    ]
    return any(keyword in query.lower() for keyword in generic_keywords)


def update_image_paths(md_file, prefix_path, workspace_logger):
    """
    Update image paths in a Markdown file by adding a prefix.
    """
    try:
        workspace_logger.info("Updating image paths...")
        with open(md_file, "r", encoding="utf-8") as file:
            md_content = file.read()

        updated_md_content = re.sub(
            r'!\[Image\]\((.*?)\)',
            lambda m: f"![Image]({prefix_path}/{m.group(1)})",
            md_content
        )

        with open(md_file, "w", encoding="utf-8") as file:
            file.write(updated_md_content)

        workspace_logger.info("Image paths updated successfully.")
    except Exception as e:
        workspace_logger.error(f"Error updating image paths: {e}")
        raise