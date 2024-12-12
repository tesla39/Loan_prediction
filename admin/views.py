from django.shortcuts import render,redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User

# Create your views here.
from django.contrib.auth.models import User
from django.contrib.auth import authenticate,login as log
from django.contrib.auth import logout
from django.contrib import messages
from user.models import LoanModel

# Create your views here.


def login(request):
    return render(request,'admin/login.html')

@login_required(login_url='/admin')
def index(request):
    return render(request,'admin/adminindex.html')

@login_required(login_url='/admin')
def contact(request):
    return render(request,'admin/contact.html')

def login_superuser(request):

        username= request.POST['username']
        password= request.POST['password']

        user = authenticate(request,username=username, password=password)

        
        if user is not None:
            log(request,user)

            if user.is_superuser == 1:

                return redirect('/admin/home')

            else:
                messages.error(request,"Username and Password Don't Match, Please Try Again !")

                return redirect('/admin/login')

        else:
            messages.error(request,"Username and Password Don't Match, Please Try Again !")
           
            return redirect('/admin')
    
def log_out(request):
    logout(request)
    return redirect('/admin')

@login_required(login_url='/admin')
def details(request):
    user=User.objects.all() 
    return render(request,'admin/admindetails.html',{'user':user})

@login_required(login_url='/admin')
def delete_user(request,id):
    user=User.objects.get(id=id)
    user.delete()
    user=User.objects.all()
    return render(request,'admin/admindetails.html',{'user':user})

@login_required(login_url='/admin')
def delete_data(request,user_id):
   data=LoanModel.objects.get(user_id=user_id)
   data.delete()
   data=LoanModel.objects.all()
   return render(request,'admin/loan.html',{'loans':data})


def loan(request):
    loans=LoanModel.objects.all()
    return render(request,'admin/loan.html',{'loans':loans})
