from fastapi import HTTPException, status


class BaseAPIException(HTTPException):
    """Clase base para todas las excepciones de nuestra API"""

class NotFoundException(BaseAPIException):
    def __init__(self, detail: str="Resource not found"):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)

class ConflictException(BaseAPIException):
    def __init__(self, detail: str="Resource conflict"):
        super().__init__(status_code=status.HTTP_409_CONFLICT, detail=detail)

class ValidationException(BaseAPIException):
    def __init__(self, detail: str="Validation error"):
        super().__init__(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail)