import asyncio
from functools import partial
import logging
import traceback
from pathlib import Path
from aiohttp import web
from concurrent.futures import ThreadPoolExecutor  # or ProcessPoolExecutor
from backend.resampler import Resampler
from config import CONFIG

server_ready = False  # /GET Use


def split_arguments(input_string: str):
    otherargs = input_string.split(" ")[-11:]
    file_path_strings = " ".join(input_string.split(" ")[:-11])
    first_file, second_file = file_path_strings.split(".wav ")
    return [first_file + ".wav", second_file] + otherargs


async def handle_get(request: web.Request):
    if server_ready:
        return web.Response(text="Server Ready", status=200)
    else:
        return web.Response(text="Server Initializing", status=503)


async def handle_post(request: web.Request):
    global server_ready

    if not server_ready:
        logging.warning("POST arrived but server not ready.")
        return web.Response(text="Server initializing, please retry.", status=503)

    post_data_string = await request.text()
    logging.info(f"post_data_string: {post_data_string}")

    try:
        sliced = split_arguments(post_data_string)
        in_file_path, out_file_path = Path(sliced[0]), Path(sliced[1])
        note = f"'{in_file_path.stem}' -> '{out_file_path.name}'"
        logging.info(f"Queued {note} ...")

        loop = asyncio.get_running_loop()
        job = partial(Resampler, *sliced)
        await loop.run_in_executor(infer_executor, job)

        logging.info(f"Processing {note} successful.")
        return web.Response(text=f"Success: {note}", status=200)

    except FileNotFoundError:
        err = "Error processing: Input file not found."
        logging.error(err, exc_info=True)
        return web.Response(text=f"{err}\n{traceback.format_exc()}", status=404)

    except Exception:
        err = "Error processing: Internal error."
        logging.error(err, exc_info=True)
        return web.Response(text=f"{err}\n{traceback.format_exc()}", status=500)


def run(port: int = 8572):
    global server_ready, infer_executor
    infer_executor = ThreadPoolExecutor(
        max_workers=CONFIG.max_workers
    )  # Inference Task Queue

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
    )

    app = web.Application()
    app.add_routes([web.get("/", handle_get), web.post("/", handle_post)])

    server_ready = True
    logging.info(
        f"Listening on {port}; aiohttp + inference-thread={CONFIG.max_workers}"
    )
    web.run_app(
        app, port=port, access_log=None
    )  # Production environment uses gunicorn -k aiohttp.worker.GunicornWebWorker ...


if __name__ == "__main__":
    run()
