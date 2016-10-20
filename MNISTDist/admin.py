from django.contrib import admin
from .models import Device,MiniBatch,Epoch

admin.site.register(Device)
admin.site.register(MiniBatch)
admin.site.register(Epoch)
# Register your models here.
