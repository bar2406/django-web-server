from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^home', views.home, name='home'),
    url(r'^imalive', views.imalive, name='imalive'),
    url(r'^getdataset', views.getdataset, name='getdataset'),
    url(r'^sendresults', views.sendresults, name='sendresults'),

]