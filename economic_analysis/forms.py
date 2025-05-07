from django import forms
import json

class ProjectForm(forms.Form):
    budget = forms.FloatField(label='Начальный бюджет', min_value=0)
    projects = forms.CharField(widget=forms.Textarea, label='Проекты (формат JSON)')

    def clean_projects(self):
        # Чистим строку, чтобы удалить лишние пробелы
        projects_str = self.cleaned_data.get('projects').strip()

        try:
            # Пытаемся загрузить строку как JSON
            projects = json.loads(projects_str)
        except json.JSONDecodeError:
            raise forms.ValidationError("Ошибка в формате JSON. Убедитесь, что данные проекта являются правильным JSON.")

        # Если проекты пустые, можно добавить валидацию
        if not projects:
            raise forms.ValidationError("Необходимо указать хотя бы один проект.")

        return projects
