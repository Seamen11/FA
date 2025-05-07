import base64
from io import BytesIO
import matplotlib.pyplot as plt
from django.http import HttpResponse
from django.shortcuts import render, redirect
from .forms import ProjectForm
import json
from .model_core import GlobalEconomy, CompanyEconomy, State, InvestmentModel


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
    return render(request, 'home.html')

def projects_page(request):
    if request.method == 'POST':
        form = ProjectForm(request.POST)

        if form.is_valid():
            # Получаем значение бюджета из формы
            budget = form.cleaned_data['budget']

            # Создаем объекты GlobalEconomy и CompanyEconomy
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

            # Инициализируем начальное состояние
            initial_state = State(
                budget=budget,
                market_condition="neutral",  # начальные условия рынка
                global_econ=global_econ,
                company_econ=company_econ
            )

            # Получаем данные проектов из формы
            projects_str = form.cleaned_data['projects']
            try:
                projects = json.loads(projects_str) if isinstance(projects_str, str) else projects_str
            except json.JSONDecodeError:
                return HttpResponse("Ошибка в формате JSON", status=400)

            # Создаем инвестиционную модель
            investment_model = InvestmentModel(projects=projects, initial_state=initial_state)
            max_profit, strategy, states, econ_history = investment_model.optimize()

            # Генерация графика
            plot_data = plot_to_base64(states, econ_history)

            # Передаем данные в шаблон для отображения результатов
            result = {
                'max_profit': max_profit,
                'strategy': strategy,
                'states': states,
                'econ_history': econ_history,
                'plot_data': plot_data,
                'budget': budget,
                'gdp': global_econ.gdp,
                'interest_rate': global_econ.interest_rate,
                'exchange_rate': global_econ.exchange_rate,
                'oil_price': global_econ.oil_price,
                'inflation': global_econ.inflation,
                'sanctions': global_econ.sanctions,
                'market_condition': initial_state.market_condition,
                'projects': projects
            }

            return render(request, 'calculate.html', {'result': result})
    else:
        form = ProjectForm()

    return render(request, 'projects.html', {'form': form})

def enter_projects_page(request):
    if request.method == 'POST':
        # Получаем начальный бюджет
        budget = float(request.POST.get('budget', 300))  # Начальный бюджет
        print(f"Начальный бюджет: {budget}")

        # Инициализация глобальной и корпоративной экономики с дефолтными значениями
        global_econ = GlobalEconomy(
            gdp=100.0,  # Значение по умолчанию для ВВП
            interest_rate=0.05,  # Значение по умолчанию для процентной ставки
            exchange_rate=1.0,  # Значение по умолчанию для курса обмена
            oil_price=50.0,  # Значение по умолчанию для цены нефти
            inflation=0.03,  # Значение по умолчанию для инфляции
            sanctions=False  # Значение по умолчанию для санкций
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
            budget=budget,  # Используем полученное значение бюджета
            market_condition="neutral",  # начальные условия рынка
            global_econ=global_econ,
            company_econ=company_econ
        )

        # Получаем данные проектов из формы
        projects_data = []
        i = 1
        while f'project_{i}_levels' in request.POST and f'project_{i}_profits' in request.POST:
            levels_str = request.POST.get(f'project_{i}_levels', '')
            profits_str = request.POST.get(f'project_{i}_profits', '')

            if levels_str and profits_str:
                try:
                    levels = list(map(int, levels_str.split(',')))  # Преобразуем строки в список чисел
                    profits = list(map(int, profits_str.split(',')))
                    projects_data.append({'levels': levels, 'profits': profits})
                except ValueError:
                    return HttpResponse("Ошибка в формате данных для уровней инвестиций или прибыли", status=400)

            i += 1

        if not projects_data:
            return HttpResponse("Не указаны проекты для расчета.", status=400)

        # Создаем инвестиционную модель с полученными проектами
        investment_model = InvestmentModel(projects=projects_data, initial_state=initial_state)

        # Запускаем оптимизацию
        max_profit, strategy, states, econ_history = investment_model.optimize()

        # Генерация графика
        plot_data = plot_to_base64(states, econ_history)

        # Передаем данные в шаблон
        result = {
            'max_profit': max_profit,
            'strategy': strategy,
            'states': states,
            'econ_history': econ_history,
            'plot_data': plot_data,
            'budget': budget,
            'gdp': global_econ.gdp,
            'interest_rate': global_econ.interest_rate,
            'exchange_rate': global_econ.exchange_rate,
            'oil_price': global_econ.oil_price,
            'inflation': global_econ.inflation,
            'sanctions': global_econ.sanctions,
            'market_condition': initial_state.market_condition,
        }

        return render(request, 'calculate.html', {'result': result})

    # Если GET-запрос, просто рендерим страницу с текущими данными
    return render(request, 'enter_projects.html')


def parameters_page(request):
    result = None
    if request.method == 'POST':
        budget = float(request.POST.get('budget', 300))
        print(f"Полученный начальный бюджет: {budget}")
        # Получаем параметры из формы
        gdp = float(request.POST.get('gdp', 100.0))
        interest_rate = float(request.POST.get('interest_rate', 0.05))
        exchange_rate = float(request.POST.get('exchange_rate', 1.0))
        oil_price = float(request.POST.get('oil_price', 50.0))
        inflation = float(request.POST.get('inflation', 0.03))
        sanctions = request.POST.get('sanctions') == 'True'
        market_condition = request.POST.get('market_condition', 'neutral')

        # Получаем начальный бюджет из формы
        print(f"Полученный начальный бюджет: {budget}")

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

        # Используем переданный начальный бюджет
        initial_state = State(
            budget=budget,
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
        budget = float(request.POST.get('budget', 300))
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
            budget=budget,
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
            'budget': budget,
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

