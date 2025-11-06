import requests
from PIL import Image
import io

# create two small images
img1 = Image.new('RGB', (100,100), color=(255,255,255))
img2 = Image.new('RGB', (100,100), color=(250,250,250))

buf1 = io.BytesIO()
buf2 = io.BytesIO()
img1.save(buf1, format='PNG')
img2.save(buf2, format='PNG')
buf1.seek(0)
buf2.seek(0)

files = {'image1': ('a.png', buf1, 'image/png'), 'image2': ('b.png', buf2, 'image/png')}

try:
    r = requests.post('http://127.0.0.1:5000/compare', files=files)
    print('Status:', r.status_code)
    print('Response:', r.text)
except Exception as e:
    print('Error:', e)
