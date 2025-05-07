from django.shortcuts import render, redirect
from .model_core import InvestmentModel, GlobalEconomy, CompanyEconomy, State, Action

# Функция обработки формы и запуска модели
def home(request):
    result = None
    if request.method == 'POST':
        form_data = request.POST

        # Параметры, которые пользователь вводит на странице
        use_defaults = form_data.get("use_defaults") == "on"  # Проверка для галочки "Использовать встроенные данные"
        include_shocks = form_data.get("include_shocks") == "on"  # Проверка для галочки "Включить экономические шоки"
        include_taxes = form_data.get("include_taxes") == "on"  # Проверка для галочки "Учитывать налоги"

        # Если пользователь выбрал использовать встроенные данные
        if use_defaults:
            global_econ = GlobalEconomy()
            global_econ.oil_price_vol = 0.5
            global_econ.exchange_vol = 0.3
            global_econ.gdp_growth_vol = 0.05

            company_econ = CompanyEconomy(global_economy=global_econ)
            initial_state = State(
                budget=300,
                market_condition="neutral",
                global_econ=global_econ,
                company_econ=company_econ
            )

            # Если включены шоки, определяем их последовательность
            shock_sequence = ["neutral", "rate_hike", "oil_crisis", "financial_crisis", "neutral"] if include_shocks else []

            # Задаем проекты для расчета
            projects = [
                {'levels': [0, 50, 100, 150, 200, 250, 300],
                 'profits': [0, 20, 44, 60, 75, 85, 95]},
                {'levels': [0, 50, 100, 150, 200, 250, 300],
                 'profits': [0, 5, 51, 70, 80, 90, 100]},
                {'levels': [0, 50, 100, 150, 200, 250, 300],
                 'profits': [0, 10, 50, 65, 75, 85, 90]},
                {'levels': [0, 50, 100, 150, 200, 250, 300],
                 'profits': [0, 8, 44, 60, 72, 85, 90]},
                {'levels': [0, 50, 100, 150, 200, 250, 300],
                 'profits': [0, 16, 39, 55, 76, 92, 99]}
            ]
        else:
            # Если не использует встроенные данные, нужно реализовать ручной ввод
            # Для простоты, это пока можно оставить на заглушке
            raise NotImplementedError("Поддержка ручного ввода будет добавлена позже.")

        # Запуск инвестиционной модели с выбранными проектами и шоками
        model = InvestmentModel(projects=projects, step=50, initial_state=initial_state, shock_sequence=shock_sequence)
        max_profit, strategy, states, econ_history = model.optimize()

        # Создаем отладочный лог для отображения
        debug_log = []
        for s in states:
            debug_log.append(
                f"[DEBUG] Этап {s['step']}: вложено {s['invested']}, накопленная прибыль {s['profit']:.2f}, остаток бюджета {s['remaining_budget']:.2f}")

        # Формируем результат для отображения
        result = {
            'max_profit': round(max_profit, 2),
            'strategy': strategy,
            'states': states,
            'econ_history': econ_history,
            'shock_sequence': shock_sequence,
            'debug_log': "\n".join(debug_log)
        }

        # После расчета перенаправляем на страницу с результатами
        return render(request, 'result.html', {'result': result})

    return render(request, 'home.html')
