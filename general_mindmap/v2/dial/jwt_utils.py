from time import time

import jwt
from fastapi import HTTPException, Request


def get_expiration(token: str) -> float:
    decoded_payload = jwt.decode(token, options={"verify_signature": False})

    return decoded_payload["exp"]


def check_expiration(token: str) -> bool:
    return get_expiration(token) >= time() + 5 * 60


def check_jwt(request: Request):
    pass
    # if "authorization" in request.headers and not check_expiration(
    #     request.headers["authorization"].split()[1]
    # ):
    #     raise HTTPException(status_code=401, detail="JWT is too old")
