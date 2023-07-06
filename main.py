import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

import vnd_log
from routers import health_route, socket_stream_route
from dconfig import config_object

app = FastAPI()

app.mount("/static", StaticFiles(directory="./web/static"), name="static")


@app.get("/")
async def get():
    return {"message": "Car detect"}


app.include_router(health_route.router)
app.include_router(socket_stream_route.router)

if __name__ == '__main__':
    uvicorn.run(app, host=config_object.SERVER_NAME, port=config_object.PORT_NUMBER, proxy_headers=True)
    vnd_log.dlog_i(f"Server started at {config_object.SERVER_NAME}:{config_object.PORT_NUMBER}")
