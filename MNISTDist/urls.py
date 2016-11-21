from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^imalive', views.imalive, name='imalive'),
	url(r'^getNeuralNet', views.getNeuralNet, name='getNeuralNet'),
	url(r'^getTrainSet', views.getTrainSet, name='getTrainSet'),
	url(r'^getTestSet', views.getTestSet, name='getTestSet'),
    url(r'^getData', views.getData, name='getData'),
    url(r'^postData', views.postData, name='postData'),

]