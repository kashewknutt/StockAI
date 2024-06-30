from django.shortcuts import render, redirect
from django.core.mail import send_mail
from .forms import ContactForm

def contact(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            # Process the form data
            name = form.cleaned_data['name']
            user_email = form.cleaned_data['email']
            message = form.cleaned_data['message']

            # Construct the confirmation email message
            confirmation_message = f"""
            Dear {name},

            Thank you for connecting to StockAI. We have received the following message from you and will respond to your query as soon as possible:

            "{message}"

            Best regards,
            StockAI Team
            """

            # Send the confirmation email to the user
            send_mail(
                subject="StockAI - We Received Your Message",
                message=confirmation_message,
                from_email='stock.kashewknutt@gmail.com',  # Your verified Gmail address
                recipient_list=[user_email],  # User's email
                fail_silently=False,
            )

            return redirect('contact_success')
    else:
        form = ContactForm()
    return render(request, 'contact.html', {'form': form})

def success(request):
    return render(request, 'contact_success.html')