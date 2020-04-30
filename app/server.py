from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import sys
import uvicorn, aiohttp, asyncio
from io import BytesIO
from PIL import Image

from tensorflow.keras.models import load_model
import cv2
import numpy as np
import pathlib

export_file_url = 'https://www.dropbox.com/s/kupg33j3ygn4o0r/covid-19_2x2.model?dl=1'
export_file_name = 'covid-19_2x2.model'

classes = ['Infected', 'Normal']
path = pathlib.Path().parent.absolute()

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if pathlib.Path(dest).exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, str(path/export_file_name))
    try:
        learn = load_model(str(path/export_file_name))
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nSome thing went wrong"
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'app/view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = Image.open(BytesIO(img_bytes))
    img = img.convert('RGB')
    img = img.resize((224, 224)) 
    img = np.array(img)
    data = np.array([img]) / 255.0  
    prediction = learn.predict(data)[0]
    return JSONResponse({'result': str(classes[np.argmax(prediction)])})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
