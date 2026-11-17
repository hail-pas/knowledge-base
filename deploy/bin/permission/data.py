"""Optional initial menu/resource definitions.

API permissions are synchronized from protected FastAPI routes. MenuAndPerm
can be populated with menu trees that bind those permission codes to resources.
"""

MenuAndPerm: list[dict] = []
