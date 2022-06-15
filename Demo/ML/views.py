from django.shortcuts import redirect, render
from .md import KNN
from django.core.files.storage import FileSystemStorage
# Create your views here.
def home(request):
    return render(request,"index.html")

def predict(request):
    try:
        fileObj = request.FILES['imgpath']
        fs = FileSystemStorage()
        fs.save(fileObj.name,fileObj)
        path = ("./media/"+fileObj.name)
        a = KNN(path)
        return render(request, "index.html", {'class':a,'img':path})
    except:
        path = ("./static/cheems.jpg")
        return render(request, "index.html", {'class':'','img':path})

        