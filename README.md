# django-web-server

work done:
creat four main pages, corruspanding to four main functions:
  1.recive new device:
		need to update to real dataset URL
  2.send device dataset to process:
		need to complete two functions: getSubsetData() and getNeuralNet()
		need to figure out how to send all the data we want-in npz file? other format? string?
		need to figure out how we even save neuralNet-as chainer object or simply numpy matrix
  3.get results and stats from device
		no work was done
  4.display ui with stats about whatever
		no work was done



whats left on server side:
  1.understend how to send meta-data and not a simple http page
  2.understand how to recive meta-data from device and how to process it
  3.think of a smart way to send training set. we cant create a link specificlly for each device (each device has a uniqeu minibatch)
  4.how to get this server up and running in LAN
  5.get the server up and running on the internet
 
whats left on device side:
  EVERYTHING, dont know where to start from
  
whats left on chainer and DL side:
  1.how to split dataset in a smart way, how to create own dataset and oterators
  2.mimshak between chainer npz files and python-how to open, process (avarage) and save
