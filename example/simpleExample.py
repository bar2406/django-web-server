import requests as rq
url = "http://127.0.0.1:8000/MNIST"
model = "my_model"
p = rq.post(url + r"/imalive", data=model)
devId = p.text
npzFile = rq.post(url + r"/getData", data=devId)
with open(r"bla.npz", 'wb') as fd:
	for chunk in p.iter_content(10):
		fd.write(chunk)