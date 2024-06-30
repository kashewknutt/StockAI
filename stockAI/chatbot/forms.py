# forms.py

from django import forms

class StockQueryForm(forms.Form):
    query = forms.CharField(label='Enter a Stock Symbol', max_length=10)
