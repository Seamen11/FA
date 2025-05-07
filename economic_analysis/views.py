import base64
from io import BytesIO
import matplotlib.pyplot as plt
from django.http import HttpResponse
from django.shortcuts import render, redirect
import json

from .forms import ProjectForm
from .model_core import GlobalEconomy, CompanyEconomy, State, InvestmentModel  # Ваши классы



def plot_to_base64(states, econ_history):
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    steps = [s['step'] for s in states]
    profits = [s['profit'] for s in states]
    budgets = [s['remaining_budget'] for s in states]

    inflation = [e['inflation'] for e in econ_history]
    interest = [e['interest_rate'] for e in econ_history]
    oil = [e['oil_price'] for e in econ_history]
    gdp = [e['gdp'] for e in econ_history]

    axs[0].plot(steps, profits, marker='o', label="Накопленная прибыль")
    axs[0].plot(steps, budgets, marker='s', label="Оставшийся бюджет")
    axs[0].set_title("Финансовая динамика")
    axs[0].set_ylabel("Значение")
    axs[0].legend()

    axs[1].plot(steps, inflation, marker='o', label="Инфляция")
    axs[1].plot(steps, interest, marker='x', label="Процентная ставка")
    axs[1].plot(steps, oil, marker='s', label="Цена нефти")
    axs[1].plot(steps, gdp, marker='^', label="ВВП")
    axs[1].set_title("Экономическая динамика")
    axs[1].set_xlabel("Этап")
    axs[1].set_ylabel("Значение")
    axs[1].legend()

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return plot_data


def home(request):
    result = None
    projects = [
        {'levels': [0, 50, 100, 150, 200, 250, 300], 'profits': [0, 20, 44, 60, 75, 85, 95]},
        {'levels': [0, 50, 100, 150, 200, 250, 300], 'profits': [0,  5, 51, 70, 80, 90, 100]},
        {'levels': [0, 50, 100, 150, 200, 250, 300], 'profits': [0, 10, 50, 65, 75, 85, 90]},
        {'levels': [0, 50, 100, 150, 200, 250, 300], 'profits': [0,  8, 44, 60, 72, 85, 90]},
        {'levels': [0, 50, 100, 150, 200, 250, 300], 'profits': [0, 16, 39, 55, 76, 92, 99]}
    ]

    result = None  # Изначально результат пустой
    if request.method == "POST":
        # Выполняем расчёты только при POST-запросе
        # Здесь будет код для расчётов вашей инвестиционной модели
        global_econ = GlobalEconomy(
            gdp=100.0,
            interest_rate=0.05,
            exchange_rate=1.0,
            oil_price=50.0,
            inflation=0.03,
            sanctions=False
        )
        # 💡 Добавим последовательность шоков и усилим волатильность
        shock_sequence = ["neutral", "rate_hike", "oil_crisis", "financial_crisis", "neutral"]
        global_econ.oil_price_vol = 0.5
        global_econ.exchange_vol = 0.3
        global_econ.gdp_growth_vol = 0.05

        print("Волатильности:")
        print({
            "oil_price_vol": global_econ.oil_price_vol,
            "exchange_vol": global_econ.exchange_vol,
            "gdp_growth_vol": global_econ.gdp_growth_vol
        })

        company_econ = CompanyEconomy(
            debt=0.0,
            amortization_type='linear',
            accumulated_loss=0.0,
            loan_term=5,
            global_economy=global_econ,
            dividend_percentage=0.05
        )
        initial_state = State(
            budget=300,
            market_condition="neutral",
            global_econ=global_econ,
            company_econ=company_econ
        )

        model = InvestmentModel(projects, step=50, initial_state=initial_state)
        max_profit, strategy, states, econ_history = model.optimize()

        # Генерация графиков
        plot_data = plot_to_base64(states, econ_history)

        result = {
            'max_profit': max_profit,
            'strategy': strategy,
            'states': states,
            'econ_history': econ_history,
            'plot_data': plot_data
        }

        return render(request, 'home.html', {'result': result})

    # Если GET-запрос, просто рендерим страницу с пустыми результатами
    return render(request, 'home.html', {'result': result, 'projects': projects})


def projects_page(request):
    if request.method == 'POST':
        form = ProjectForm(request.POST)

        if form.is_valid():
            # Создаем объект GlobalEconomy
            global_econ = GlobalEconomy(
                gdp=100.0,
                interest_rate=0.05,
                exchange_rate=1.0,
                oil_price=50.0,
                inflation=0.03,
                sanctions=False
            )

            # Создаем объект CompanyEconomy
            company_econ = CompanyEconomy(
                debt=0.0,
                amortization_type='linear',
                accumulated_loss=0.0,
                loan_term=5,
                global_economy=global_econ,
                dividend_percentage=0.05
            )

            # Инициализируем начальное состояние
            initial_state = State(
                budget=300,  # начальный бюджет
                market_condition="neutral",  # начальные условия рынка
                global_econ=global_econ,
                company_econ=company_econ
            )

            # Получаем данные проектов из формы
            projects_str = form.cleaned_data['projects']
            print("Проекты из формы:", projects_str)  # Для отладки

            # Парсим строку JSON в список словарей
            try:
                projects = json.loads(projects_str)
                print("Проекты после парсинга:", projects)
            except json.JSONDecodeError:
                return HttpResponse("Ошибка в формате JSON", status=400)

            # Создаем инвестиционную модель
            investment_model = InvestmentModel(projects=projects, initial_state=initial_state)

            # Запускаем оптимизацию
            max_profit, strategy, states, econ_history = investment_model.optimize()

            # Генерация графика
            plot_data = plot_to_base64(states, econ_history)

            # Передаем данные в шаблон
            return render(request, 'calculate.html', {
                'max_profit': max_profit,
                'strategy': strategy,
                'states': states,
                'econ_history': econ_history,
                'plot_data': plot_data  # Передаем график в шаблон
            })
    else:
        form = ProjectForm()

    return render(request, 'projects.html', {'form': form})


def calculate(request):
    return HttpResponse("Calculation result page.")