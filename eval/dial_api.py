import asyncio
import io
import json
from typing import Any, Dict, Tuple
from uuid import uuid4

import aiohttp
import requests

from config import DIAL_API_KEY, DIAL_URL


def auth_headers() -> Dict[str, str]:
    return {"API-KEY": DIAL_API_KEY}


def get_bucket() -> str:
    print("Get bucket")

    response = requests.get(f"{DIAL_URL}/v1/bucket", headers=auth_headers())

    return response.json()["bucket"]


def read_state() -> Tuple[Dict[str, Any], str]:
    print("Read state")

    bucket = get_bucket()

    response = requests.get(
        f"{DIAL_URL}/v1/files/{bucket}/state.json", headers=auth_headers()
    )

    if response.status_code == 200:
        return response.json(), response.headers["ETag"]
    else:
        return {}, ""


def write_state(state: Dict[str, Any], etag: str):
    print("Send write request")

    bucket = get_bucket()

    response = requests.put(
        f"{DIAL_URL}/v1/files/{bucket}/state.json",
        files={
            "file": ("state.json", io.StringIO(json.dumps(state)), "application/json")
        },
        headers=auth_headers() | {"if-match": etag},
    )

    return response.status_code


async def create_app():
    bucket = get_bucket()

    id = uuid4()

    async with aiohttp.ClientSession() as session:
        async with session.put(
            f"{DIAL_URL}/v1/applications/{bucket}/{id}",
            headers=auth_headers(),
            json={
                "application_type_schema_id": "http://dev-dial-core.staging.deltixhub.io/custom_application_schemas/mindmapapps",
                "application_properties": {},
            },
        ) as response:
            return (await response.json())["url"] if response.status == 200 else ""


async def add_source(app_id: str, source_bytes: bytes):
    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field(
            "file",
            source_bytes,
            filename="source.pdf",
            content_type="application/pdf",
        )

        async with session.post(
            f"{DIAL_URL}/v1/deployments/{app_id}/route/v1/sources",
            data=form,
            headers=auth_headers(),
        ) as response:
            return await response.text()


async def get_graph(app_id: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{DIAL_URL}/v1/deployments/{app_id}/route/v1/graph",
            headers=auth_headers(),
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                print(f"[{app_id}] Can't get the graph. Status: {response.status}")
                return None


async def export(app_id: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"{DIAL_URL}/v1/deployments/{app_id}/route/v1/sources/export",
            headers=auth_headers(),
        ) as response:
            response.raise_for_status()

            return await response.read()


async def import_state(app_id: str, state: bytes):
    async with aiohttp.ClientSession() as session:
        form = aiohttp.FormData()
        form.add_field(
            "file",
            state,
            filename="mind_map.zip",
            content_type="application/zip",
        )

        async with session.post(
            f"{DIAL_URL}/v1/deployments/{app_id}/route/v1/import",
            data=form,
            headers=auth_headers(),
        ) as response:
            response.raise_for_status()

            return await response.text()


async def run_generation(app_id: str):
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(
                total=None,
                sock_read=None,
            )
        ) as session:
            async with session.post(
                f"{DIAL_URL}/v1/deployments/{app_id}/route/v1/generate",
                headers=auth_headers(),
            ) as response:
                async for line in response.content:
                    print(f"[{app_id}] {line}", flush=True)
    except aiohttp.ClientPayloadError:
        print(f"[{app_id}] Client payload error", flush=True)


async def set_params(app_id: str, model: str, prompt: str, rag_prompt: str):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{DIAL_URL}/v1/deployments/{app_id}/route/v1/generate/params",
            headers=auth_headers(),
            json={
                "type": "lite",
                "prompt": prompt,
                "rag_prompt": rag_prompt,
                "model": model,
            },
        ) as response:
            return response.status
