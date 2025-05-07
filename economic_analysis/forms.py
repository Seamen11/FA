from django import forms

class StrategyForm(forms.Form):
    use_defaults = forms.BooleanField(required=False, label="Использовать встроенные данные")
    budget = forms.FloatField(initial=300, required=False)
    market_condition = forms.ChoiceField(choices=[('good', 'Хорошее'), ('neutral', 'Нейтральное'), ('bad', 'Плохое')], required=False)
    include_shocks = forms.BooleanField(required=False, label="Учитывать экономические шоки?")
