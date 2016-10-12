import datetime
from django.db import models
from django.utils.encoding import python_2_unicode_compatible
from django.utils import timezone
# Create your models here.

@python_2_unicode_compatible  # only if you need to support Python 2
class device(models.Model):
    #device stats
    ID=models.IntegerField()    #unique device ID
    devicemodel = models.CharField(max_length=200)  #device model i.e "Sony Xperia Z3 Compact"
    connection_time = models.DateTimeField()   #time of reciveving "im alive" signal from this device
    lastactivetime=models.DateTimeField()   #last time this device reqeusted data.
    totaldatasetsgiven=models.IntegerField()    #some way of marking how much work this device has done
    
    #last minibatch data
    minibatchID=models.IntegerField()   #some ID of the current minibatch this device is working on. not sure its actually an int
    epoch=models.IntegerField() #from what epoch this minibatch is from


    def __str__(self):
        return self.question_text
    def was_published_recently(self):
        return self.pub_date >= timezone.now() - datetime.timedelta(days=1)