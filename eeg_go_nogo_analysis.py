# Скрипт для анализа данных ЭЭГ эксперимента GO/NoGo
# Сравниваем две группы: высокая мотивация vs низкая мотивация
#
# Анализируем компоненты ERP:
# - N2 (190-350 мс) - отрицательный пик, связан с торможением
# - P2 (100-250 мс) - положительный пик, привлечение внимания
# - P3 (500-700 мс) - средняя амплитуда, распределение ресурсов
#
# Используем электроды средней линии (Fz, Cz, Pz и т.д.)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Настройки для графиков - чтобы выглядели нормально
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

# ============================================================================
# Настройки - здесь можно менять параметры анализа
# ============================================================================

# Разделение на группы по опроснику мотивации
HIGH_MOTIVATION = ['participant1', 'participant4', 'participant7', 'participant8']
LOW_MOTIVATION = ['participant2', 'participant3', 'participant5', 'participant6']

# Какие электроды использовать для анализа
# В идеале хотелось бы все пять, но у нас могут быть не все
ELECTRODES_PREFERRED = ['Fz', 'FCz', 'Cz', 'CPz', 'Pz']
# Если основных нет, используем хотя бы эти три
ELECTRODES_ALTERNATIVE = ['Fz', 'Cz', 'Pz']

# Временные окна для поиска компонентов (в миллисекундах)
# Взял из статьи Staub et al. 2014
COMPONENT_WINDOWS = {
    'N2': (190, 350),    # здесь ищем отрицательный пик N2
    'P2': (100, 250),    # здесь ищем положительный пик P2
    'P3': (500, 700),    # здесь измеряем среднюю амплитуду P3
}

# Вокруг найденного пика усредняем ±15 мс (как в статье)
PEAK_WINDOW = 15

# Базовый период для коррекции - от -200 до 0 мс (до появления стимула)
BASELINE_WINDOW = (-200, 0)

# ============================================================================
# Функции для загрузки данных
# ============================================================================

def load_eeg_data(participant_folder, condition='GO'):
    """
    Загружает данные ЭЭГ для одного испытуемого.
    condition может быть 'GO' или 'NOGO'
    """
    folder_path = Path(participant_folder)
    
    # Файлы с _1.xlsx - это GO стимул, с _2.xlsx - NoGo
    if condition == 'GO':
        files = list(folder_path.glob('*_1.xlsx'))
    else:
        files = list(folder_path.glob('*_2.xlsx'))
    
    if not files:
        raise FileNotFoundError(f"Не найден файл для {condition} в {participant_folder}")
    
    file_path = files[0]
    print(f"  Загрузка {condition}: {file_path.name}")
    
    # Читаем Excel файл
    df = pd.read_excel(file_path)
    
    # Иногда колонка времени может называться по-другому, проверяем
    if 'Time (ms)' not in df.columns:
        if df.columns[0].lower().startswith('time') or df.columns[0] == 'Unnamed: 0':
            df = df.rename(columns={df.columns[0]: 'Time (ms)'})
    
    return df


def load_all_participants(group_participants, condition='GO'):
    """
    Загружает данные для всех испытуемых в группе.
    Возвращает словарь где ключ - имя участника, значение - его данные.
    """
    all_data = {}
    
    for participant in group_participants:
        try:
            data = load_eeg_data(participant, condition)
            all_data[participant] = data
        except Exception as e:
            # Если не получилось загрузить - пропускаем, но выводим ошибку
            print(f"Ошибка при загрузке {participant} ({condition}): {e}")
            continue
    
    return all_data


# ============================================================================
# Функции для обработки данных
# ============================================================================

def baseline_correction(data, time_col='Time (ms)', baseline_window=BASELINE_WINDOW):
    """
    Делаем коррекцию базовой линии - вычитаем среднее значение до стимула.
    Это стандартная процедура в анализе ERP, чтобы убрать дрейф сигнала.
    """
    corrected_data = data.copy()
    time = corrected_data[time_col].values
    
    # Находим где у нас базовый период (до стимула)
    baseline_mask = (time >= baseline_window[0]) & (time <= baseline_window[1])
    
    # Для каждого электрода вычитаем среднее значение базового периода
    electrode_cols = [col for col in corrected_data.columns if col != time_col]
    
    for col in electrode_cols:
        if col in corrected_data.columns:
            # Считаем среднее в базовом периоде и вычитаем его из всего сигнала
            baseline_mean = corrected_data.loc[baseline_mask, col].mean()
            corrected_data[col] = corrected_data[col] - baseline_mean
    
    return corrected_data


def interpolate_time(data_dict, time_col='Time (ms)'):
    """
    Приводит данные всех испытуемых к одной временной сетке.
    Нужно потому что у разных участников могут быть немного разные временные точки.
    """
    # Сначала находим общий диапазон времени у всех
    all_times = []
    for data in data_dict.values():
        all_times.extend(data[time_col].values)
    
    min_time = min(all_times)
    max_time = max(all_times)
    
    # Создаем единую временную сетку с шагом 1 мс
    common_time = np.arange(min_time, max_time + 1, 1)
    
    # Теперь для каждого испытуемого интерполируем данные на эту сетку
    interpolated_dict = {}
    electrode_cols = [col for col in list(data_dict.values())[0].columns if col != time_col]
    
    for participant, data in data_dict.items():
        interpolated_data = pd.DataFrame()
        interpolated_data[time_col] = common_time
        
        for col in electrode_cols:
            if col in data.columns:
                # Простая линейная интерполяция
                interpolated_values = np.interp(
                    common_time,
                    data[time_col].values,
                    data[col].values
                )
                interpolated_data[col] = interpolated_values
        
        interpolated_dict[participant] = interpolated_data
    
    return interpolated_dict, common_time


def average_across_participants(data_dict, time_col='Time (ms)'):
    """
    Усредняет данные всех испытуемых в группе.
    Сначала приводит всех к одной временной сетке, потом усредняет.
    """
    # Сначала интерполируем всех на одну временную сетку
    interpolated_dict, common_time = interpolate_time(data_dict, time_col)
    
    # Теперь усредняем по всем испытуемым
    electrode_cols = [col for col in list(interpolated_dict.values())[0].columns if col != time_col]
    
    averaged_data = pd.DataFrame()
    averaged_data[time_col] = common_time
    
    for col in electrode_cols:
        # Берем значения этого электрода у всех испытуемых
        all_values = np.array([df[col].values for df in interpolated_dict.values()])
        # И просто усредняем
        averaged_data[col] = np.mean(all_values, axis=0)
    
    return averaged_data


# ============================================================================
# Функции для анализа ERP компонентов
# ============================================================================

def find_peak_in_window(data, time, electrode, window, peak_type='negative'):
    """
    Ищет пик в заданном временном окне.
    peak_type: 'negative' для N2 (ищем минимум), 'positive' для P2 (ищем максимум)
    Возвращает латентность пика, его амплитуду и среднюю амплитуду вокруг пика.
    """
    # Вырезаем нужное временное окно
    mask = (time >= window[0]) & (time <= window[1])
    window_time = time[mask]
    window_data = data[electrode].values[mask]
    
    if len(window_data) == 0:
        return np.nan, np.nan, np.nan
    
    # Ищем пик - минимум для N2, максимум для P2
    if peak_type == 'negative':
        peak_idx = np.argmin(window_data)
    else:
        peak_idx = np.argmax(window_data)
    
    peak_latency = window_time[peak_idx]
    peak_amplitude = window_data[peak_idx]
    
    # Усредняем в окне ±15 мс вокруг найденного пика (как в статье)
    peak_window_start = peak_latency - PEAK_WINDOW
    peak_window_end = peak_latency + PEAK_WINDOW
    
    peak_mask = (time >= peak_window_start) & (time <= peak_window_end)
    mean_amplitude = np.mean(data[electrode].values[peak_mask])
    
    return peak_latency, peak_amplitude, mean_amplitude


def measure_p3_amplitude(data, time, electrode, window):
    """
    Для P3 не ищем пик, а просто считаем среднюю амплитуду в окне 500-700 мс.
    Так делают потому что P3 часто широкий и не имеет четкого пика.
    """
    mask = (time >= window[0]) & (time <= window[1])
    mean_amplitude = np.mean(data[electrode].values[mask])
    return mean_amplitude


def get_available_electrodes(data, preferred=ELECTRODES_PREFERRED, alternative=ELECTRODES_ALTERNATIVE):
    """
    Проверяет какие электроды есть в данных.
    Сначала ищем предпочтительные, если их нет - используем альтернативные.
    """
    available = []
    for electrode in preferred:
        if electrode in data.columns:
            available.append(electrode)
    
    # Если вообще ничего не нашли, пробуем альтернативные
    if len(available) == 0:
        for electrode in alternative:
            if electrode in data.columns:
                available.append(electrode)
    
    return available


def analyze_erp_components(data, time_col='Time (ms)'):
    """
    Основная функция анализа - ищет все компоненты (N2, P2, P3) для всех электродов.
    Возвращает словарь с результатами.
    """
    time = data[time_col].values
    results = {}
    
    # Сначала определяем какие электроды у нас есть
    available_electrodes = get_available_electrodes(data)
    
    for electrode in available_electrodes:
        if electrode not in data.columns:
            print(f"Предупреждение: электрод {electrode} не найден в данных")
            continue
        
        electrode_results = {}
        
        # Ищем N2 - отрицательный пик в окне 190-350 мс
        n2_latency, n2_peak, n2_mean = find_peak_in_window(
            data, time, electrode, COMPONENT_WINDOWS['N2'], peak_type='negative'
        )
        electrode_results['N2'] = {
            'latency': n2_latency,
            'peak_amplitude': n2_peak,
            'mean_amplitude': n2_mean
        }
        
        # Ищем P2 - положительный пик в окне 100-250 мс
        p2_latency, p2_peak, p2_mean = find_peak_in_window(
            data, time, electrode, COMPONENT_WINDOWS['P2'], peak_type='positive'
        )
        electrode_results['P2'] = {
            'latency': p2_latency,
            'peak_amplitude': p2_peak,
            'mean_amplitude': p2_mean
        }
        
        # Для P3 просто считаем среднюю амплитуду в окне 500-700 мс
        p3_mean = measure_p3_amplitude(
            data, time, electrode, COMPONENT_WINDOWS['P3']
        )
        electrode_results['P3'] = {
            'mean_amplitude': p3_mean
        }
        
        results[electrode] = electrode_results
    
    return results


def analyze_individual_participants(data_dict, time_col='Time (ms)'):
    """
    Анализирует компоненты для каждого испытуемого отдельно.
    Нужно для статистики - чтобы потом сравнивать группы.
    """
    individual_results = {}
    
    for participant, data in data_dict.items():
        # Сначала делаем коррекцию базовой линии
        corrected_data = baseline_correction(data, time_col)
        
        # Потом анализируем компоненты
        results = analyze_erp_components(corrected_data, time_col)
        individual_results[participant] = results
    
    return individual_results


# ============================================================================
# Статистический анализ
# ============================================================================

def mann_whitney_u_test(high_group_values, low_group_values):
    """
    Делает тест Манна-Уитни для сравнения двух групп.
    Это непараметрический тест, подходит для малых выборок.
    """
    # Убираем пропущенные значения (NaN)
    high_clean = [v for v in high_group_values if not np.isnan(v)]
    low_clean = [v for v in low_group_values if not np.isnan(v)]
    
    # Нужно хотя бы по 2 значения в каждой группе
    if len(high_clean) < 2 or len(low_clean) < 2:
        return np.nan, np.nan
    
    statistic, p_value = stats.mannwhitneyu(high_clean, low_clean, alternative='two-sided')
    return statistic, p_value


def perform_statistical_tests(high_results, low_results):
    """
    Сравнивает две группы по всем компонентам и электродам.
    Для каждого сравнения делает тест Манна-Уитни и считает средние/стандартные отклонения.
    """
    stats_results = {}
    
    components = ['N2', 'P2', 'P3']
    measures = {
        'N2': ['mean_amplitude', 'latency'],
        'P2': ['mean_amplitude', 'latency'],
        'P3': ['mean_amplitude']
    }
    
    # Сначала собираем все электроды которые есть в данных
    all_electrodes = set()
    for participant_results in list(high_results.values()) + list(low_results.values()):
        all_electrodes.update(participant_results.keys())
    all_electrodes = sorted(list(all_electrodes))
    
    for component in components:
        stats_results[component] = {}
        
        for measure in measures[component]:
            stats_results[component][measure] = {}
            
            for electrode in all_electrodes:
                # Собираем значения этого компонента у всех испытуемых в группе
                high_values = []
                low_values = []
                
                # Группа с высокой мотивацией
                for participant in HIGH_MOTIVATION:
                    if participant in high_results:
                        if electrode in high_results[participant]:
                            if component in high_results[participant][electrode]:
                                value = high_results[participant][electrode][component].get(measure)
                                if value is not None:
                                    high_values.append(value)
                
                # Группа с низкой мотивацией
                for participant in LOW_MOTIVATION:
                    if participant in low_results:
                        if electrode in low_results[participant]:
                            if component in low_results[participant][electrode]:
                                value = low_results[participant][electrode][component].get(measure)
                                if value is not None:
                                    low_values.append(value)
                
                # Если есть данные в обеих группах - делаем тест
                if len(high_values) > 0 and len(low_values) > 0:
                    statistic, p_value = mann_whitney_u_test(high_values, low_values)
                    stats_results[component][measure][electrode] = {
                        'statistic': statistic,
                        'p_value': p_value,
                        'high_mean': np.mean(high_values),
                        'low_mean': np.mean(low_values),
                        'high_std': np.std(high_values),
                        'low_std': np.std(low_values)
                    }
    
    return stats_results


# ============================================================================
# Функции для построения графиков
# ============================================================================

def plot_grand_average_waveforms(high_go, high_nogo, low_go, low_nogo, time_col='Time (ms)'):
    """
    Рисует усредненные ERP волны для всех групп и условий.
    На одном графике видно GO и NoGo для обеих групп.
    """
    time = high_go[time_col].values
    
    # Определяем какие электроды у нас есть
    available_electrodes = get_available_electrodes(high_go)
    n_electrodes = len(available_electrodes)
    
    # Увеличиваем размер графика и делаем больше места между подграфиками
    fig, axes = plt.subplots(n_electrodes, 1, figsize=(18, 5*n_electrodes))
    if n_electrodes == 1:
        axes = [axes]
    
    for idx, electrode in enumerate(available_electrodes):
        ax = axes[idx]
        
        if electrode not in high_go.columns:
            ax.text(0.5, 0.5, f'Электрод {electrode} не найден', 
                   transform=ax.transAxes, ha='center')
            continue
        
        # Рисуем все четыре линии: GO и NoGo для обеих групп
        ax.plot(time, high_go[electrode], label='Высокая мотивация - GO', 
               linewidth=2.5, color='blue', alpha=0.8)
        ax.plot(time, high_nogo[electrode], label='Высокая мотивация - NoGo', 
               linewidth=2.5, color='red', alpha=0.8, linestyle='--')
        ax.plot(time, low_go[electrode], label='Низкая мотивация - GO', 
               linewidth=2.5, color='green', alpha=0.8)
        ax.plot(time, low_nogo[electrode], label='Низкая мотивация - NoGo', 
               linewidth=2.5, color='orange', alpha=0.8, linestyle='--')
        
        # Линия на 0 мс - момент появления стимула
        ax.axvline(x=0, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
        
        # Линия на 0 мкВ для ориентира
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.4)
        
        # Закрашиваем окна где ищем компоненты (для наглядности)
        ax.axvspan(COMPONENT_WINDOWS['N2'][0], COMPONENT_WINDOWS['N2'][1], 
                  alpha=0.08, color='purple')
        ax.axvspan(COMPONENT_WINDOWS['P2'][0], COMPONENT_WINDOWS['P2'][1], 
                  alpha=0.08, color='cyan')
        ax.axvspan(COMPONENT_WINDOWS['P3'][0], COMPONENT_WINDOWS['P3'][1], 
                  alpha=0.08, color='yellow')
        
        # Подписи для окон компонентов (только на первом графике)
        if idx == 0:
            ax.text((COMPONENT_WINDOWS['N2'][0] + COMPONENT_WINDOWS['N2'][1])/2, 
                   ax.get_ylim()[1]*0.9, 'N2', ha='center', fontsize=10, 
                   color='purple', alpha=0.7)
            ax.text((COMPONENT_WINDOWS['P2'][0] + COMPONENT_WINDOWS['P2'][1])/2, 
                   ax.get_ylim()[1]*0.9, 'P2', ha='center', fontsize=10, 
                   color='cyan', alpha=0.7)
            ax.text((COMPONENT_WINDOWS['P3'][0] + COMPONENT_WINDOWS['P3'][1])/2, 
                   ax.get_ylim()[1]*0.9, 'P3', ha='center', fontsize=10, 
                   color='orange', alpha=0.7)
        
        ax.set_xlabel('Время (мс)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Амплитуда (мкВ)', fontsize=13, fontweight='bold')
        ax.set_title(f'Электрод {electrode} - Усредненные ERP волны', 
                    fontsize=14, fontweight='bold', pad=15)
        
        # Размещаем легенду так чтобы не налазила на график
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9, 
                 ncol=2, columnspacing=0.8)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Увеличиваем размер шрифта на осях
        ax.tick_params(labelsize=11)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('grand_average_waveforms.png', dpi=300, bbox_inches='tight')
    print("График сохранен: grand_average_waveforms.png")
    plt.close()


def plot_component_comparison(stats_results):
    """
    Рисует столбчатые диаграммы для сравнения амплитуд компонентов между группами.
    Звездочки показывают статистическую значимость различий.
    """
    components = ['N2', 'P2', 'P3']
    measures = {
        'N2': 'mean_amplitude',
        'P2': 'mean_amplitude',
        'P3': 'mean_amplitude'
    }
    
    # Увеличиваем размер графика
    fig, axes = plt.subplots(1, len(components), figsize=(20, 7))
    if len(components) == 1:
        axes = [axes]
    
    for idx, component in enumerate(components):
        ax = axes[idx]
        measure = measures[component]
        
        electrodes = []
        high_means = []
        low_means = []
        high_stds = []
        low_stds = []
        p_values = []
        
        # Собираем данные для всех электродов
        for electrode in sorted(stats_results[component][measure].keys()):
            data = stats_results[component][measure][electrode]
            electrodes.append(electrode)
            high_means.append(data['high_mean'])
            low_means.append(data['low_mean'])
            high_stds.append(data['high_std'])
            low_stds.append(data['low_std'])
            p_values.append(data['p_value'])
        
        if len(electrodes) == 0:
            continue
        
        x = np.arange(len(electrodes))
        width = 0.4
        
        # Рисуем столбцы для обеих групп
        bars1 = ax.bar(x - width/2, high_means, width, yerr=high_stds, 
                      label='Высокая мотивация', alpha=0.85, color='blue', 
                      capsize=8, error_kw={'elinewidth': 2})
        bars2 = ax.bar(x + width/2, low_means, width, yerr=low_stds, 
                      label='Низкая мотивация', alpha=0.85, color='orange', 
                      capsize=8, error_kw={'elinewidth': 2})
        
        # Добавляем звездочки если есть значимые различия
        # Вычисляем максимальную высоту для правильного размещения
        max_height = max([h + s for h, s in zip(high_means + low_means, 
                                                high_stds + low_stds)])
        
        for i, p_val in enumerate(p_values):
            y_pos = max(high_means[i] + high_stds[i], low_means[i] + low_stds[i]) + abs(max_height)*0.1
            if p_val < 0.001:
                ax.text(i, y_pos, '***', ha='center', fontsize=16, 
                       fontweight='bold', color='red')
            elif p_val < 0.01:
                ax.text(i, y_pos, '**', ha='center', fontsize=16, 
                       fontweight='bold', color='red')
            elif p_val < 0.05:
                ax.text(i, y_pos, '*', ha='center', fontsize=16, 
                       fontweight='bold', color='red')
        
        ax.set_xlabel('Электроды', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'{component} - Средняя амплитуда (мкВ)', 
                     fontsize=13, fontweight='bold')
        ax.set_title(f'Сравнение {component} между группами', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(electrodes, fontsize=12)
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.tick_params(labelsize=11)
        
        # Добавляем горизонтальную линию на 0 для ориентира
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    
    plt.tight_layout(pad=2.0)
    plt.savefig('component_comparison.png', dpi=300, bbox_inches='tight')
    print("График сохранен: component_comparison.png")
    plt.close()


def plot_latency_comparison(stats_results):
    """
    Рисует сравнение латентностей компонентов (когда появляется пик).
    Для P3 латентность не считаем, только для N2 и P2.
    """
    components = ['N2', 'P2']
    
    # Увеличиваем размер графика
    fig, axes = plt.subplots(1, len(components), figsize=(16, 7))
    if len(components) == 1:
        axes = [axes]
    
    for idx, component in enumerate(components):
        ax = axes[idx]
        
        electrodes = []
        high_means = []
        low_means = []
        high_stds = []
        low_stds = []
        p_values = []
        
        for electrode in sorted(stats_results[component]['latency'].keys()):
            data = stats_results[component]['latency'][electrode]
            electrodes.append(electrode)
            high_means.append(data['high_mean'])
            low_means.append(data['low_mean'])
            high_stds.append(data['high_std'])
            low_stds.append(data['low_std'])
            p_values.append(data['p_value'])
        
        if len(electrodes) == 0:
            continue
        
        x = np.arange(len(electrodes))
        width = 0.4
        
        bars1 = ax.bar(x - width/2, high_means, width, yerr=high_stds, 
                      label='Высокая мотивация', alpha=0.85, color='blue', 
                      capsize=8, error_kw={'elinewidth': 2})
        bars2 = ax.bar(x + width/2, low_means, width, yerr=low_stds, 
                      label='Низкая мотивация', alpha=0.85, color='orange', 
                      capsize=8, error_kw={'elinewidth': 2})
        
        # Звездочки если есть значимые различия
        max_height = max([h + s for h, s in zip(high_means + low_means, 
                                                high_stds + low_stds)])
        for i, p_val in enumerate(p_values):
            if p_val < 0.05:
                y_pos = max(high_means[i] + high_stds[i], low_means[i] + low_stds[i]) + abs(max_height)*0.08
                ax.text(i, y_pos, '*', ha='center', fontsize=16, 
                       fontweight='bold', color='red')
        
        ax.set_xlabel('Электроды', fontsize=13, fontweight='bold')
        ax.set_ylabel(f'{component} - Латентность (мс)', 
                     fontsize=13, fontweight='bold')
        ax.set_title(f'Латентность {component} между группами', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(electrodes, fontsize=12)
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.tick_params(labelsize=11)
    
    plt.tight_layout(pad=2.0)
    plt.savefig('latency_comparison.png', dpi=300, bbox_inches='tight')
    print("График сохранен: latency_comparison.png")
    plt.close()


def plot_go_nogo_comparison(high_go, high_nogo, low_go, low_nogo, time_col='Time (ms)'):
    """
    Рисует сравнение GO vs NoGo для каждой группы отдельно.
    Полезно чтобы увидеть эффект торможения в каждой группе.
    """
    time = high_go[time_col].values
    available_electrodes = get_available_electrodes(high_go)
    n_electrodes = len(available_electrodes)
    
    # Два ряда графиков - один для высокой мотивации, другой для низкой
    fig, axes = plt.subplots(2, n_electrodes, figsize=(6*n_electrodes, 10))
    if n_electrodes == 1:
        axes = axes.reshape(2, 1)
    
    # Верхний ряд - высокая мотивация
    for idx, electrode in enumerate(available_electrodes):
        ax = axes[0, idx]
        
        if electrode not in high_go.columns:
            continue
        
        ax.plot(time, high_go[electrode], label='GO', 
               linewidth=2.5, color='blue', alpha=0.8)
        ax.plot(time, high_nogo[electrode], label='NoGo', 
               linewidth=2.5, color='red', alpha=0.8, linestyle='--')
        
        ax.axvline(x=0, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.4)
        
        ax.set_title(f'{electrode} - Высокая мотивация', 
                    fontsize=13, fontweight='bold', pad=10)
        ax.set_ylabel('Амплитуда (мкВ)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=10)
    
    # Нижний ряд - низкая мотивация
    for idx, electrode in enumerate(available_electrodes):
        ax = axes[1, idx]
        
        if electrode not in low_go.columns:
            continue
        
        ax.plot(time, low_go[electrode], label='GO', 
               linewidth=2.5, color='green', alpha=0.8)
        ax.plot(time, low_nogo[electrode], label='NoGo', 
               linewidth=2.5, color='orange', alpha=0.8, linestyle='--')
        
        ax.axvline(x=0, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
        ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.4)
        
        ax.set_title(f'{electrode} - Низкая мотивация', 
                    fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Время (мс)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Амплитуда (мкВ)', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=10)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('go_nogo_comparison.png', dpi=300, bbox_inches='tight')
    print("График сохранен: go_nogo_comparison.png")
    plt.close()


def plot_component_details(stats_results, high_go_avg, high_nogo_avg, 
                          low_go_avg, low_nogo_avg, time_col='Time (ms)'):
    """
    Рисует детальные графики для каждого компонента отдельно.
    Показывает волны и столбчатые диаграммы вместе.
    """
    components = ['N2', 'P2', 'P3']
    available_electrodes = get_available_electrodes(high_go_avg)
    time = high_go_avg[time_col].values
    
    for component in components:
        measure = 'mean_amplitude'
        
        # Проверяем есть ли данные для этого компонента
        if component not in stats_results or measure not in stats_results[component]:
            continue
        
        n_electrodes = len(available_electrodes)
        fig, axes = plt.subplots(2, n_electrodes, figsize=(6*n_electrodes, 10))
        if n_electrodes == 1:
            axes = axes.reshape(2, 1)
        
        # Верхний ряд - волны
        for idx, electrode in enumerate(available_electrodes):
            ax = axes[0, idx]
            
            if electrode not in high_go_avg.columns:
                continue
            
            # Рисуем волны в окне компонента
            window = COMPONENT_WINDOWS[component]
            mask = (time >= window[0] - 50) & (time <= window[1] + 50)
            window_time = time[mask]
            
            # Рисуем волны в окне компонента
            ax.plot(window_time, high_go_avg[electrode].values[mask], 
                   label='Высокая - GO', linewidth=2, color='blue', alpha=0.7)
            ax.plot(window_time, high_nogo_avg[electrode].values[mask], 
                   label='Высокая - NoGo', linewidth=2, color='red', alpha=0.7, linestyle='--')
            ax.plot(window_time, low_go_avg[electrode].values[mask], 
                   label='Низкая - GO', linewidth=2, color='green', alpha=0.7)
            ax.plot(window_time, low_nogo_avg[electrode].values[mask], 
                   label='Низкая - NoGo', linewidth=2, color='orange', alpha=0.7, linestyle='--')
            
            # Закрашиваем окно компонента
            ax.axvspan(window[0], window[1], alpha=0.15, color='gray')
            ax.axvline(x=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
            ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.3)
            
            ax.set_title(f'{electrode} - {component} волны', 
                        fontsize=12, fontweight='bold', pad=10)
            ax.set_ylabel('Амплитуда (мкВ)', fontsize=11)
            ax.legend(fontsize=9, loc='best', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.tick_params(labelsize=9)
        
        # Нижний ряд - столбчатые диаграммы
        for idx, electrode in enumerate(available_electrodes):
            ax = axes[1, idx]
            
            if electrode not in stats_results[component][measure]:
                continue
            
            data = stats_results[component][measure][electrode]
            
            # Столбцы для GO и NoGo (если есть данные)
            categories = ['Высокая\nмотивация', 'Низкая\nмотивация']
            means = [data['high_mean'], data['low_mean']]
            stds = [data['high_std'], data['low_std']]
            
            x = np.arange(len(categories))
            width = 0.6
            
            bars = ax.bar(x, means, width, yerr=stds, alpha=0.8, 
                         color=['blue', 'orange'], capsize=8, 
                         error_kw={'elinewidth': 2})
            
            # Добавляем значение на столбцы
            for i, (m, s) in enumerate(zip(means, stds)):
                ax.text(i, m + s + abs(max(means))*0.05, f'{m:.2f}', 
                       ha='center', fontsize=11, fontweight='bold')
            
            # Звездочка если значимо
            p_val = data['p_value']
            if p_val < 0.05:
                max_height = max([m + s for m, s in zip(means, stds)])
                ax.text(0.5, max_height + abs(max(means))*0.15, 
                       '*' if p_val >= 0.01 else '**' if p_val >= 0.001 else '***',
                       ha='center', fontsize=14, fontweight='bold', color='red',
                       transform=ax.get_xaxis_transform())
            
            ax.set_ylabel(f'{component} амплитуда (мкВ)', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(categories, fontsize=11)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
            ax.tick_params(labelsize=10)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(f'{component}_detailed.png', dpi=300, bbox_inches='tight')
        print(f"График сохранен: {component}_detailed.png")
        plt.close()


def plot_group_differences(high_go, high_nogo, low_go, low_nogo, time_col='Time (ms)'):
    """
    Рисует график разностей между группами (высокая - низкая мотивация).
    Полезно чтобы увидеть где именно группы различаются.
    """
    time = high_go[time_col].values
    available_electrodes = get_available_electrodes(high_go)
    n_electrodes = len(available_electrodes)
    
    fig, axes = plt.subplots(n_electrodes, 1, figsize=(18, 5*n_electrodes))
    if n_electrodes == 1:
        axes = [axes]
    
    for idx, electrode in enumerate(available_electrodes):
        ax = axes[idx]
        
        if electrode not in high_go.columns:
            continue
        
        # Вычисляем разности: высокая - низкая мотивация
        diff_go = high_go[electrode] - low_go[electrode]
        diff_nogo = high_nogo[electrode] - low_nogo[electrode]
        
        ax.plot(time, diff_go, label='Разность GO (Высокая - Низкая)', 
               linewidth=2.5, color='blue', alpha=0.8)
        ax.plot(time, diff_nogo, label='Разность NoGo (Высокая - Низкая)', 
               linewidth=2.5, color='red', alpha=0.8, linestyle='--')
        
        # Линия на 0 - где нет различий
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        ax.axvline(x=0, color='black', linestyle=':', linewidth=1.5, alpha=0.6)
        
        # Закрашиваем окна компонентов
        ax.axvspan(COMPONENT_WINDOWS['N2'][0], COMPONENT_WINDOWS['N2'][1], 
                  alpha=0.08, color='purple')
        ax.axvspan(COMPONENT_WINDOWS['P2'][0], COMPONENT_WINDOWS['P2'][1], 
                  alpha=0.08, color='cyan')
        ax.axvspan(COMPONENT_WINDOWS['P3'][0], COMPONENT_WINDOWS['P3'][1], 
                  alpha=0.08, color='yellow')
        
        ax.set_xlabel('Время (мс)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Разность амплитуд (мкВ)', fontsize=13, fontweight='bold')
        ax.set_title(f'Электрод {electrode} - Разности между группами', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(labelsize=11)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('group_differences.png', dpi=300, bbox_inches='tight')
    print("График сохранен: group_differences.png")
    plt.close()


def save_statistics_table(stats_results, filename='statistics_results.txt'):
    """
    Сохраняет все результаты статистики в текстовый файл.
    Удобно для отчета - можно скопировать таблицы.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("РЕЗУЛЬТАТЫ СТАТИСТИЧЕСКОГО АНАЛИЗА (Тест Манна-Уитни)\n")
        f.write("=" * 80 + "\n\n")
        
        components = ['N2', 'P2', 'P3']
        measures = {
            'N2': ['mean_amplitude', 'latency'],
            'P2': ['mean_amplitude', 'latency'],
            'P3': ['mean_amplitude']
        }
        
        for component in components:
            f.write(f"\n{'='*80}\n")
            f.write(f"КОМПОНЕНТ: {component}\n")
            f.write(f"{'='*80}\n\n")
            
            for measure in measures[component]:
                f.write(f"  Измерение: {measure}\n")
                f.write(f"  {'-'*76}\n")
                f.write(f"  {'Электрод':<12} {'Высокая (M±SD)':<20} {'Низкая (M±SD)':<20} {'U':<10} {'p':<10}\n")
                f.write(f"  {'-'*76}\n")
                
                # Собираем все доступные электроды
                all_electrodes = sorted(stats_results[component][measure].keys())
                
                for electrode in all_electrodes:
                    data = stats_results[component][measure][electrode]
                    high_str = f"{data['high_mean']:.2f}±{data['high_std']:.2f}"
                    low_str = f"{data['low_mean']:.2f}±{data['low_std']:.2f}"
                    p_val = data['p_value']
                    p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
                    
                    # Добавляем звездочки для значимости
                    if p_val < 0.001:
                        p_str += " ***"
                    elif p_val < 0.01:
                        p_str += " **"
                    elif p_val < 0.05:
                        p_str += " *"
                    
                    f.write(f"  {electrode:<12} {high_str:<20} {low_str:<20} {data['statistic']:<10.2f} {p_str:<10}\n")
                
                f.write("\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Примечание: * p<0.05, ** p<0.01, *** p<0.001\n")
        f.write("="*80 + "\n")
    
    print(f"Результаты статистики сохранены: {filename}")


# ============================================================================
# Главная функция - здесь все запускается
# ============================================================================

def main():
    """
    Основная функция - запускает весь анализ от начала до конца.
    """
    print("="*80)
    print("АНАЛИЗ ДАННЫХ ЭЭГ ЭКСПЕРИМЕНТА GO/NOGO")
    print("="*80)
    print("\nГруппы испытуемых:")
    print(f"  Высокая мотивация: {', '.join(HIGH_MOTIVATION)}")
    print(f"  Низкая мотивация: {', '.join(LOW_MOTIVATION)}")
    print("\n" + "="*80 + "\n")
    
    # ========================================================================
    # Шаг 1: Загружаем данные
    # ========================================================================
    print("ШАГ 1: Загрузка данных...")
    
    print("\nЗагрузка данных группы с высокой мотивацией:")
    high_go_data = load_all_participants(HIGH_MOTIVATION, condition='GO')
    high_nogo_data = load_all_participants(HIGH_MOTIVATION, condition='NOGO')
    
    print("\nЗагрузка данных группы с низкой мотивацией:")
    low_go_data = load_all_participants(LOW_MOTIVATION, condition='GO')
    low_nogo_data = load_all_participants(LOW_MOTIVATION, condition='NOGO')
    
    print(f"\nЗагружено:")
    print(f"  Высокая мотивация - GO: {len(high_go_data)} испытуемых")
    print(f"  Высокая мотивация - NoGo: {len(high_nogo_data)} испытуемых")
    print(f"  Низкая мотивация - GO: {len(low_go_data)} испытуемых")
    print(f"  Низкая мотивация - NoGo: {len(low_nogo_data)} испытуемых")
    
    # ========================================================================
    # Шаг 2: Обрабатываем данные
    # ========================================================================
    print("\n" + "="*80)
    print("ШАГ 2: Обработка данных...")
    
    # Делаем коррекцию базовой линии для каждого испытуемого отдельно
    print("\nКоррекция базовой линии...")
    for participant in list(high_go_data.keys()):
        high_go_data[participant] = baseline_correction(high_go_data[participant])
        high_nogo_data[participant] = baseline_correction(high_nogo_data[participant])
    
    for participant in list(low_go_data.keys()):
        low_go_data[participant] = baseline_correction(low_go_data[participant])
        low_nogo_data[participant] = baseline_correction(low_nogo_data[participant])
    
    # Теперь усредняем по группам - получаем grand average
    print("\nУсреднение данных по группам...")
    high_go_avg = average_across_participants(high_go_data)
    high_nogo_avg = average_across_participants(high_nogo_data)
    low_go_avg = average_across_participants(low_go_data)
    low_nogo_avg = average_across_participants(low_nogo_data)
    
    print("  Усреднение завершено")
    
    # ========================================================================
    # Шаг 3: Анализируем ERP компоненты
    # ========================================================================
    print("\n" + "="*80)
    print("ШАГ 3: Анализ ERP компонентов...")
    
    # Сначала анализируем усредненные данные - для графиков
    print("\nАнализ усредненных данных...")
    high_go_results = analyze_erp_components(high_go_avg)
    high_nogo_results = analyze_erp_components(high_nogo_avg)
    low_go_results = analyze_erp_components(low_go_avg)
    low_nogo_results = analyze_erp_components(low_nogo_avg)
    
    # Потом анализируем каждого испытуемого отдельно - для статистики
    print("\nАнализ индивидуальных данных...")
    high_go_individual = analyze_individual_participants(high_go_data)
    high_nogo_individual = analyze_individual_participants(high_nogo_data)
    low_go_individual = analyze_individual_participants(low_go_data)
    low_nogo_individual = analyze_individual_participants(low_nogo_data)
    
    # Для статистики используем данные GO (можно добавить NoGo отдельно если нужно)
    print("\nОбъединение результатов GO и NoGo...")
    high_all_individual = {}
    low_all_individual = {}
    
    for participant in high_go_individual.keys():
        high_all_individual[participant] = high_go_individual[participant]
    
    for participant in low_go_individual.keys():
        low_all_individual[participant] = low_go_individual[participant]
    
    # ========================================================================
    # Шаг 4: Статистика
    # ========================================================================
    print("\n" + "="*80)
    print("ШАГ 4: Статистический анализ (тест Манна-Уитни)...")
    
    stats_results = perform_statistical_tests(high_all_individual, low_all_individual)
    print("  Статистический анализ завершен")
    
    # ========================================================================
    # Шаг 5: Рисуем графики
    # ========================================================================
    print("\n" + "="*80)
    print("ШАГ 5: Создание графиков...")
    
    print("\nПостроение усредненных волн...")
    plot_grand_average_waveforms(high_go_avg, high_nogo_avg, 
                                 low_go_avg, low_nogo_avg)
    
    print("\nПостроение сравнения компонентов...")
    plot_component_comparison(stats_results)
    
    print("\nПостроение сравнения латентностей...")
    plot_latency_comparison(stats_results)
    
    print("\nПостроение сравнения GO vs NoGo...")
    plot_go_nogo_comparison(high_go_avg, high_nogo_avg, low_go_avg, low_nogo_avg)
    
    print("\nПостроение детальных графиков компонентов...")
    plot_component_details(stats_results, high_go_avg, high_nogo_avg, 
                          low_go_avg, low_nogo_avg)
    
    print("\nПостроение графиков разностей между группами...")
    plot_group_differences(high_go_avg, high_nogo_avg, low_go_avg, low_nogo_avg)
    
    # ========================================================================
    # Шаг 6: Сохраняем результаты
    # ========================================================================
    print("\n" + "="*80)
    print("ШАГ 6: Сохранение результатов...")
    
    save_statistics_table(stats_results)
    
    # Еще сохраняем в CSV для удобства
    print("\nСохранение численных результатов...")
    results_summary = []
    
    for component in ['N2', 'P2', 'P3']:
        for measure in ['mean_amplitude', 'latency'] if component != 'P3' else ['mean_amplitude']:
            if component in stats_results and measure in stats_results[component]:
                for electrode in sorted(stats_results[component][measure].keys()):
                    data = stats_results[component][measure][electrode]
                    results_summary.append({
                        'Component': component,
                        'Measure': measure,
                        'Electrode': electrode,
                        'High_Mean': data['high_mean'],
                        'High_STD': data['high_std'],
                        'Low_Mean': data['low_mean'],
                        'Low_STD': data['low_std'],
                        'U_Statistic': data['statistic'],
                        'P_Value': data['p_value']
                    })
    
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv('results_summary.csv', index=False, encoding='utf-8-sig')
    print("  Результаты сохранены: results_summary.csv")
    
    # ========================================================================
    # ВЫВОД ИТОГОВ
    # ========================================================================
    print("\n" + "="*80)
    print("АНАЛИЗ ЗАВЕРШЕН!")
    print("="*80)
    print("\nСозданные файлы:")
    print("  1. grand_average_waveforms.png - Графики усредненных ERP волн")
    print("  2. component_comparison.png - Сравнение амплитуд компонентов")
    print("  3. latency_comparison.png - Сравнение латентностей компонентов")
    print("  4. go_nogo_comparison.png - Сравнение GO vs NoGo для каждой группы")
    print("  5. N2_detailed.png - Детальный график компонента N2")
    print("  6. P2_detailed.png - Детальный график компонента P2")
    print("  7. P3_detailed.png - Детальный график компонента P3")
    print("  8. group_differences.png - Разности между группами")
    print("  9. statistics_results.txt - Детальные результаты статистики")
    print("  10. results_summary.csv - Сводная таблица результатов")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

