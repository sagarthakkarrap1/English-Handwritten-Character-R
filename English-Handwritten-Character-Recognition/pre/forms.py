from dataclasses import field
from pyexpat import model
from tkinter.ttk import Label
from django import forms
from .models import Image

class ImageForm(forms.ModelForm):
    class Meta:
        model=Image
        fields = '__all__'
        labels = {'photo': ''}