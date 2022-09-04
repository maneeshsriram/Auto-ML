from django.shortcuts import redirect, render
from django.contrib.auth import logout
from django.contrib.auth.models import User
from django.contrib import messages, auth


def login(request):
    if request.method == "POST":
        user_name = request.POST['user_name']
        password = request.POST['password']
        user = auth.authenticate(username=user_name, password=password)
        if user is not None:
            auth.login(request, user)
            messages.success(request, 'Logged in')
            return redirect('home')
        else:
            messages.warning(request, 'Invalid credentials')
            return redirect('login')
    return render(request, 'accounts/login.html')


def register(request):
    if request.method == "POST":
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        user_name = request.POST['user_name']
        email = request.POST['email']
        password = request.POST['password']
        confirm_password = request.POST['confirm_password']
        if password == confirm_password:
            if User.objects.filter(username=user_name).exists():
                messages.warning(request, 'Username already exists')
                return redirect('register')
            elif User.objects.filter(email=email).exists():
                messages.warning(request, 'Email already exists')
                return redirect('register')
            else:
                user = User.objects.create_user(first_name=first_name,
                                                last_name=last_name,
                                                username=user_name,
                                                email=email,
                                                password=password
                                                )
                user.save()
                messages.success(request, "Account created successfully")
                return redirect('login')
        else:
            messages.warning(request, 'Password do not match')
            return redirect('register')
    return render(request, 'accounts/register.html')


def logout_user(request):
    logout(request)
    messages.success(request, 'Log out successful')
    return redirect('home')
