import requests as rq
url = "http://127.0.0.1:8000/MNIST"
model = "my_model"
p = rq.post(url + r"/imalive", data=model)
devId = p.text.split()[1]
npzFile = rq.post(url + r"/getData", data=devId)
with open(r"bla.npz", 'wb') as fd:
	for chunk in npzFile.iter_content(10):
		fd.write(chunk)