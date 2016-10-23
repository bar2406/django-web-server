from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^home', views.home, name='home'),
    url(r'^imalive', views.imalive, name='imalive'),
	url(r'^getNeuralNet', views.getNeuralNet, name='getNeuralNet'),
    url(r'^getData', views.getData, name='getData'),
    url(r'^postData', views.postData, name='postData'),

]