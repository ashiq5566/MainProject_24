from pyexpat import model
from tkinter.ttk import Widget
from django import forms
from django.forms import HiddenInput
from .models import Customer
from django.db.models import Q, Max, F


class CustomerForm(forms.Form):
    file = forms.FileField()
