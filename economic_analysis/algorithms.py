import numpy as np
import matplotlib.pyplot as plt
import os
from .model_core import GlobalEconomy, CompanyEconomy, State, Action, InvestmentModel

# Функция-обёртка для Django
def run_model_simulation(form_data):
    use_defaults = form_data.get("use_defaults", False)
    include_shocks = form_data.get("include_shocks", False)

    # Встроенные значения по умолчанию
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

        shock_sequence = ["neutral", "rate_hike", "oil_crisis", "financial_crisis", "neutral"] if include_shocks else []

        projects = [
            {'levels': [0, 50, 100, 150, 200, 250, 300],
             'profits': [0, 20, 44, 60, 75, 85, 95]},
            {'levels': [0, 50, 100, 150, 200, 250, 300],
             'profits': [0,  5, 51, 70, 80, 90, 100]},
            {'levels': [0, 50, 100, 150, 200, 250, 300],
             'profits': [0, 10, 50, 65, 75, 85, 90]},
            {'levels': [0, 50, 100, 150, 200, 250, 300],
             'profits': [0,  8, 44, 60, 72, 85, 90]},
            {'levels': [0, 50, 100, 150, 200, 250, 300],
             'profits': [0, 16, 39, 55, 76, 92, 99]}
        ]
    else:
        # 🧾 Альтернатива: ручной ввод из формы — добавим позже
        raise NotImplementedError("Поддержка ручного ввода будет добавлена позже.")

    model = InvestmentModel(projects=projects, step=50, initial_state=initial_state, shock_sequence=shock_sequence)
    max_profit, strategy, states, econ_history = model.optimize()

    # Отладочный лог для режима "Подробнее"
    debug_log = []
    for s in states:
        debug_log.append(f"[DEBUG] Этап {s['step']}: вложено {s['invested']}, накопленная прибыль {s['profit']:.2f}, остаток бюджета {s['remaining_budget']:.2f}")

    return {
        'max_profit': round(max_profit, 2),
        'strategy': strategy,
        'states': states,
        'econ_history': econ_history,
        'shock_sequence': shock_sequence,
        'debug_log': "\n".join(debug_log)
    }
