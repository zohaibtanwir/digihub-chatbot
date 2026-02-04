from fastapi import  Request, HTTPException

from src.dto.models import QueryRequest
from src.utils.logger import logger
from src.utils.config import BLACKLISTED_WORDS
from pydantic import BaseModel, ConfigDict

def check_blacklist(request_dict: dict, param_type: str):
    """
    Checks if any of the blacklisted words are present in the values of the request dictionary.

    Args:
        request_dict (dict): A dictionary containing key-value pairs from the request.
        param_type (str): The type of parameter being checked (e.g., 'Header', 'Request Body', 'Query Parameters').

    Raises Exception if dictionary contains blacklisted words
    """
    for key, value in request_dict.items():
        for bl_word in BLACKLISTED_WORDS:
            if (bl_word in value):
                logger.error(
                    f"Request was rejected because it contains unsanitized input {bl_word} in the {param_type}")
                raise HTTPException(status_code=400,
                                    detail="Request was rejected because it contains unsanitized input")


def validate_incoming_request(request: Request):
    """
    Validates the incoming request by checking if headers or query parameters contain blacklisted words.

    Args:
        request (Request): The incoming request object from FastAPI.

    Returns:
        None: Raises an HTTPException if any blacklisted word is found in headers or query parameters.
    """
    check_blacklist(dict(request.headers), "Header")
    check_blacklist(dict(request.query_params), "Query Parameters")


def validate_request_body(request_body: QueryRequest) -> QueryRequest:
    """
    Validates the request body by checking for any blacklisted words.

    Args:
        request_body (QueryRequest): The request body object to be validated.

    Returns:
        QueryRequest: The validated request body object.
    """
    query_dict = request_body.model_dump()
    check_blacklist(query_dict, "Request Body")
    return request_body

import time
from functools import wraps

def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # More precise than time.time()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        logger.info(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result
    return wrapper
