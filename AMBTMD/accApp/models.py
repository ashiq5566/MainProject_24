from django.db import models


class Customer(models.Model):
    file = models.FileField(null=True)