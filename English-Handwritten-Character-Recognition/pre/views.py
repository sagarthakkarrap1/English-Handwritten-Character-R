from email import charset
from urllib.request import Request
from django.http.response import HttpResponse
from django.shortcuts import render
from .forms import ImageForm
from .models import Image
from .finalsegmentation import save_lines


# Create your views here.
def index(request):
    if request.method == "POST":
        form = ImageForm(request.POST,request.FILES)
        if form.is_valid():
            print(request.FILES['photo'])
            photo = form.cleaned_data.get("photo")
            obj = Image.objects.create(
                                 photo = photo
                                 )
            obj.save()
            img_path=obj.photo.url
            pred_str=save_lines("media/"+img_path)
            
            fm = ImageForm ()
            return render(request,'index.html',{'form':fm,'obj':obj,'pred_str':pred_str})     

    fm = ImageForm ()
    return render(request,'index.html',{'form':fm}) 



