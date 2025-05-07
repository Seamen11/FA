# DjangoProject/forms.py

from django import forms

class ProjectForm(forms.Form):
    budget = forms.FloatField(label='Начальный бюджет', min_value=0)
    projects = forms.CharField(widget=forms.Textarea, label='Проекты (формат JSON)')

    # Дополнительные поля для ввода параметров проектов
    # например, уровни вложений и прибыль
