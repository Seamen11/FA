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

    # Сохраняем изображение для проверки
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.read()).decode()
    plt.close(fig)

    # Сохраняем изображение в файл для дальнейшей проверки
    with open('debug_plot.png', 'wb') as f:
        f.write(buffer.getvalue())

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


import json
from django.shortcuts import render
from .forms import ProjectForm
from .model_core import GlobalEconomy, CompanyEconomy, State, InvestmentModel  # Ваши классы


def enter_projects_page(request):
    # Начальные проекты для примера (будет заменено динамическими)
    projects = [
        {'levels': [0, 50, 100, 150, 200, 250, 300], 'profits': [0, 20, 44, 60, 75, 85, 95]},
        {'levels': [0, 50, 100, 150, 200, 250, 300], 'profits': [0, 5, 51, 70, 80, 90, 100]},
        {'levels': [0, 50, 100, 150, 200, 250, 300], 'profits': [0, 10, 50, 65, 75, 85, 90]},
        {'levels': [0, 50, 100, 150, 200, 250, 300], 'profits': [0, 8, 44, 60, 72, 85, 90]},
        {'levels': [0, 50, 100, 150, 200, 250, 300], 'profits': [0, 16, 39, 55, 76, 92, 99]}
    ]

    if request.method == 'POST':
        # Инициализация глобальной и корпоративной экономики
        global_econ = GlobalEconomy(
            gdp=100.0,
            interest_rate=0.05,
            exchange_rate=1.0,
            oil_price=50.0,
            inflation=0.03,
            sanctions=False
        )

        company_econ = CompanyEconomy(
            debt=0.0,
            amortization_type='linear',
            accumulated_loss=0.0,
            loan_term=5,
            global_economy=global_econ,
            dividend_percentage=0.05
        )

        initial_state = State(
            budget=300,  # начальный бюджет
            market_condition="neutral",  # начальные условия рынка
            global_econ=global_econ,
            company_econ=company_econ
        )

        # Получаем данные проектов из формы
        projects_data = []

        # Проходим по всем проектам, которые переданы в форму
        i = 1
        while f'project_{i}_levels' in request.POST and f'project_{i}_profits' in request.POST:
            levels = request.POST.get(f'project_{i}_levels')
            profits = request.POST.get(f'project_{i}_profits')

            if levels and profits:
                levels = list(map(int, levels.split(',')))  # Преобразуем строки в список чисел
                profits = list(map(int, profits.split(',')))
                projects_data.append({'levels': levels, 'profits': profits})

            i += 1

        # Создаем инвестиционную модель с полученными проектами
        investment_model = InvestmentModel(projects=projects_data, initial_state=initial_state)

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

    # Если GET-запрос, просто рендерим страницу с текущими данными
    return render(request, 'enter_projects.html', {'projects': projects})


def parameters_page(request):
    result = None
    if request.method == 'POST':
        # Получаем параметры из формы
        gdp = float(request.POST.get('gdp', 100.0))
        interest_rate = float(request.POST.get('interest_rate', 0.05))
        exchange_rate = float(request.POST.get('exchange_rate', 1.0))
        oil_price = float(request.POST.get('oil_price', 50.0))
        inflation = float(request.POST.get('inflation', 0.03))
        sanctions = request.POST.get('sanctions') == 'True'
        market_condition = request.POST.get('market_condition', 'neutral')

        # Создаем объекты экономики
        global_econ = GlobalEconomy(
            gdp=gdp,
            interest_rate=interest_rate,
            exchange_rate=exchange_rate,
            oil_price=oil_price,
            inflation=inflation,
            sanctions=sanctions
        )

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
            market_condition=market_condition,
            global_econ=global_econ,
            company_econ=company_econ
        )

        # Обработка проектов
        projects = []
        project_count = 1
        while f"project_{project_count}_levels" in request.POST:
            levels_str = request.POST[f"project_{project_count}_levels"]
            profits_str = request.POST[f"project_{project_count}_profits"]

            try:
                # Преобразуем строки в списки чисел
                levels = list(map(int, levels_str.split(',')))
                profits = list(map(int, profits_str.split(',')))

                # Добавляем проект в список
                projects.append({'levels': levels, 'profits': profits})
            except ValueError:
                return HttpResponse("Ошибка в формате данных проектов", status=400)

            project_count += 1

        # Если проектов не было добавлено
        if not projects:
            return HttpResponse("Проекты не были добавлены корректно", status=400)

        # Создаем модель
        investment_model = InvestmentModel(projects=projects, initial_state=initial_state)
        max_profit, strategy, states, econ_history = investment_model.optimize()

        # Генерация графика
        plot_data = plot_to_base64(states, econ_history)

        # Отправляем результат в шаблон
        result = {
            'max_profit': max_profit,
            'strategy': strategy,
            'states': states,
            'econ_history': econ_history,
            'plot_data': plot_data
        }

    return render(request, 'parameters.html', {'result': result})

def calculate_page(request):
    result = None
    if request.method == 'POST':
        # Считываем параметры из формы
        gdp = float(request.POST['gdp'])
        interest_rate = float(request.POST['interest_rate'])
        exchange_rate = float(request.POST['exchange_rate'])
        oil_price = float(request.POST['oil_price'])
        inflation = float(request.POST['inflation'])
        sanctions = request.POST['sanctions'] == 'True'
        market_condition = request.POST['market_condition']

        # Считываем проекты из формы
        projects = []
        i = 1
        while f"project_{i}_levels" in request.POST:
            levels_str = request.POST.get(f"project_{i}_levels", "").strip()
            profits_str = request.POST.get(f"project_{i}_profits", "").strip()

            if levels_str and profits_str:  # Проверка, что данные введены
                try:
                    levels = [int(level.strip()) for level in levels_str.split(",") if level.strip()]
                    profits = [int(profit.strip()) for profit in profits_str.split(",") if profit.strip()]
                    if not levels or not profits:
                        raise ValueError(f"Ошибка в данных проекта {i}")
                    projects.append({'levels': levels, 'profits': profits})
                except ValueError as e:
                    return HttpResponse(f"Ошибка в формате данных для проекта {i}. Убедитесь, что все данные числовые.", status=400)
            i += 1

        # Если проекты пустые, вернуть ошибку
        if not projects:
            return HttpResponse("Не указаны проекты для расчета.", status=400)

        # Создаем экземпляры моделей
        global_econ = GlobalEconomy(
            gdp=gdp,
            interest_rate=interest_rate,
            exchange_rate=exchange_rate,
            oil_price=oil_price,
            inflation=inflation,
            sanctions=sanctions
        )
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
            market_condition=market_condition,
            global_econ=global_econ,
            company_econ=company_econ
        )

        # Расчет инвестиционной модели
        model = InvestmentModel(projects, step=50, initial_state=initial_state)
        max_profit, strategy, states, econ_history = model.optimize()

        # Генерация графика
        plot_data = plot_to_base64(states, econ_history)

        # Передаем результаты на страницу
        result = {
            'gdp': gdp,
            'interest_rate': interest_rate,
            'exchange_rate': exchange_rate,
            'oil_price': oil_price,
            'inflation': inflation,
            'sanctions': sanctions,
            'market_condition': market_condition,
            'projects': projects,
            'max_profit': max_profit,
            'strategy': strategy,
            'plot_data': plot_data,
            'states': states,
            'econ_history': econ_history
        }

        return render(request, 'calculate.html', {'result': result})

    return render(request, 'parameters.html')

