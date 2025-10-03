import pandas as pd
import matplotlib.pyplot as plt
import os

# Проверяем существование файла
filename = "benchmark_results.csv"
if not os.path.exists(filename):
    print(f"Файл {filename} не найден!")
    print("Сначала запустите C++ бенчмарк для создания файла с результатами")
    exit(1)

# Загружаем данные
df = pd.read_csv(filename)
print("Загружены данные:")
print(df.head())
print("\nКолонки:", df.columns.tolist())

# Переименовываем колонки если нужно (приводим к ожидаемым именам)
if 'Тип' in df.columns:
    df = df.rename(columns={
        'Тип': 'type',
        'Размер матрицы': 'matrixSize', 
        'Размер блока': 'blockSize',
        'Количество потоков': 'threads',
        'Время (мс)': 'time'
    })

# Преобразуем типы данных
df['matrixSize'] = df['matrixSize'].astype(int)
df['blockSize'] = df['blockSize'].astype(int)
df['threads'] = df['threads'].astype(int)
df['time'] = df['time'].astype(float)

print("\nУникальные типы методов:", df['type'].unique())

# График 1: фиксируем блок, меняем кол-во потоков
mt_data = df[df['type'] == 'Многопоточное умножение']

if not mt_data.empty:
    fixed_block = mt_data['blockSize'].unique()[0] if len(mt_data['blockSize'].unique()) > 0 else 32
    block_data = mt_data[mt_data['blockSize'] == fixed_block]

    matrix_sizes = sorted(block_data['matrixSize'].unique())

    plt.figure(figsize=(12, 8))

    for size in matrix_sizes:
        size_data = block_data[block_data['matrixSize'] == size].sort_values('threads')
        plt.plot(size_data['threads'], size_data['time'], marker='o', label=f'Матрица {size}x{size}')

    plt.title(f"Влияние числа потоков на время (Блок {fixed_block})")
    plt.xlabel("Число потоков")
    plt.ylabel("Время выполнения (мс)")
    plt.legend(title="Размер матрицы")
    plt.grid(True)
    plt.savefig('threads_impact.png')
    plt.show()
else:
    print("Нет данных для многопоточного умножения")

# График 2: фиксируем потоки, меняем размер блока
if not mt_data.empty:
    fixed_threads = 4 if 4 in mt_data['threads'].unique() else mt_data['threads'].unique()[0]
    threads_data = mt_data[mt_data['threads'] == fixed_threads]

    matrix_sizes = sorted(threads_data['matrixSize'].unique())

    plt.figure(figsize=(12, 8))
    for size in matrix_sizes:
        size_data = threads_data[threads_data['matrixSize'] == size].sort_values('blockSize')
        plt.plot(size_data['blockSize'], size_data['time'], marker='o', linestyle='-', label=f'Матрица {size}x{size}')

    plt.title(f"Влияние размера блока на время (Потоки = {fixed_threads})")
    plt.xlabel("Размер блока")
    plt.ylabel("Время выполнения (мс)")
    plt.legend(title="Размер матрицы")
    plt.grid(True)
    plt.savefig('block_size_impact.png')
    plt.show()

# График 3: сравнение методов умножения
def best_time(data, method):
    best = []
    for size in sorted(data['matrixSize'].unique()):
        subset = data[data['matrixSize'] == size]
        if not subset.empty:
            min_time = subset['time'].min()
            best.append((size, min_time))
    return pd.DataFrame(best, columns=["matrixSize", f"time_{method}"])

# Стандартное умножение
st_data = df[df['type'] == 'Стандартное умножение']
if not st_data.empty:
    st = best_time(st_data, "std")
else:
    st = pd.DataFrame(columns=["matrixSize", "time_std"])

# Многопоточное умножение
mt_data = df[df['type'] == 'Многопоточное умножение']
if not mt_data.empty:
    mt = best_time(mt_data, "thread")
else:
    mt = pd.DataFrame(columns=["matrixSize", "time_thread"])

# Async умножение
async_data = df[df['type'] == 'Async умножение']
if not async_data.empty:
    async_res = best_time(async_data, "async")
else:
    async_res = pd.DataFrame(columns=["matrixSize", "time_async"])

# Объединяем все данные
merged = pd.DataFrame()
if not st.empty:
    merged = st
if not mt.empty:
    if merged.empty:
        merged = mt
    else:
        merged = merged.merge(mt, on="matrixSize", how='outer')
if not async_res.empty:
    if merged.empty:
        merged = async_res
    else:
        merged = merged.merge(async_res, on="matrixSize", how='outer')

if not merged.empty:
    plt.figure(figsize=(12, 8))
    
    if 'time_std' in merged.columns:
        plt.plot(merged['matrixSize'], merged['time_std'], marker='o', linewidth=2, label="Стандартное (1 поток)")
    if 'time_thread' in merged.columns:
        plt.plot(merged['matrixSize'], merged['time_thread'], marker='s', linewidth=2, label="Многопоточное (std::thread)")
    if 'time_async' in merged.columns:
        plt.plot(merged['matrixSize'], merged['time_async'], marker='^', linewidth=2, label="Async (std::async)")

    plt.title("Сравнение времени выполнения методов умножения матриц")
    plt.xlabel("Размер матрицы")
    plt.ylabel("Время выполнения (мс)")
    plt.legend()
    plt.grid(True)
    plt.savefig('methods_comparison.png')
    plt.show()

    # График 4: ускорение относительно стандартного умножения
    if 'time_std' in merged.columns and 'time_thread' in merged.columns:
        merged["speedup_thread"] = merged["time_std"] / merged["time_thread"]
    if 'time_std' in merged.columns and 'time_async' in merged.columns:
        merged["speedup_async"] = merged["time_std"] / merged["time_async"]

    plt.figure(figsize=(12, 8))
    
    if 'speedup_thread' in merged.columns:
        plt.plot(merged["matrixSize"], merged["speedup_thread"], marker="o", linewidth=2, label="Speedup std::thread")
    if 'speedup_async' in merged.columns:
        plt.plot(merged["matrixSize"], merged["speedup_async"], marker="s", linewidth=2, label="Speedup std::async")

    plt.title("Ускорение методов относительно однопоточного")
    plt.xlabel("Размер матрицы")
    plt.ylabel("Ускорение (раз)")
    plt.legend()
    plt.grid(True)
    plt.savefig('speedup_comparison.png')
    plt.show()

    # Вывод таблицы с результатами
    print("\n=== СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ ===")
    print(merged.to_string(index=False))

else:
    print("Недостаточно данных для построения графиков сравнения")

# Дополнительный анализ: лучшие параметры для каждого размера матрицы
print("\n=== ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ ===")
for method in df['type'].unique():
    method_data = df[df['type'] == method]
    if not method_data.empty:
        print(f"\n{method}:")
        for size in sorted(method_data['matrixSize'].unique()):
            size_data = method_data[method_data['matrixSize'] == size]
            best_row = size_data.loc[size_data['time'].idxmin()]
            print(f"  Размер {size}x{size}: блок={best_row['blockSize']}, потоки={best_row['threads']}, время={best_row['time']:.2f} мс")

# Дополнительные графики для ускорения и эффективности

# График 5: Ускорение от количества потоков для разных размеров матриц
if not mt_data.empty:
    plt.figure(figsize=(12, 8))
    
    # Берем оптимальный размер блока для каждого размера матрицы
    for size in sorted(mt_data['matrixSize'].unique()):
        size_data = mt_data[mt_data['matrixSize'] == size]
        
        # Находим лучший размер блока для этого размера матрицы
        best_block_for_size = size_data.loc[size_data['time'].idxmin(), 'blockSize']
        best_size_data = size_data[size_data['blockSize'] == best_block_for_size].sort_values('threads')
        
        # Находим время для 1 потока (базовое)
        single_thread_time = best_size_data[best_size_data['threads'] == 1]['time'].values
        if len(single_thread_time) > 0:
            base_time = single_thread_time[0]
            
            # Рассчитываем ускорение
            speedup_data = best_size_data.copy()
            speedup_data['speedup'] = base_time / speedup_data['time']
            
            plt.plot(speedup_data['threads'], speedup_data['speedup'], 
                    marker='o', linewidth=2, label=f'Матрица {size}x{size} (блок {best_block_for_size})')
    
    # Идеальное ускорение (линейное)
    max_threads = mt_data['threads'].max()
    ideal_threads = list(range(1, max_threads + 1))
    ideal_speedup = ideal_threads
    plt.plot(ideal_threads, ideal_speedup, 'k--', alpha=0.5, label='Идеальное ускорение')
    
    plt.title("Ускорение от количества потоков")
    plt.xlabel("Количество потоков")
    plt.ylabel("Ускорение (раз)")
    plt.legend()
    plt.grid(True)
    plt.savefig('speedup_vs_threads.png')
    plt.show()

# График 6: Эффективность от количества потоков
if not mt_data.empty:
    plt.figure(figsize=(12, 8))
    
    for size in sorted(mt_data['matrixSize'].unique()):
        size_data = mt_data[mt_data['matrixSize'] == size]
        
        # Находим лучший размер блока для этого размера матрицы
        best_block_for_size = size_data.loc[size_data['time'].idxmin(), 'blockSize']
        best_size_data = size_data[size_data['blockSize'] == best_block_for_size].sort_values('threads')
        
        # Находим время для 1 потока (базовое)
        single_thread_time = best_size_data[best_size_data['threads'] == 1]['time'].values
        if len(single_thread_time) > 0:
            base_time = single_thread_time[0]
            
            # Рассчитываем эффективность (ускорение / количество потоков)
            efficiency_data = best_size_data.copy()
            efficiency_data['efficiency'] = (base_time / efficiency_data['time']) / efficiency_data['threads'] * 100
            
            plt.plot(efficiency_data['threads'], efficiency_data['efficiency'], 
                    marker='s', linewidth=2, label=f'Матрица {size}x{size} (блок {best_block_for_size})')
    
    # Идеальная эффективность (100%)
    max_threads = mt_data['threads'].max()
    plt.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Идеальная эффективность (100%)')
    
    plt.title("Эффективность использования потоков")
    plt.xlabel("Количество потоков")
    plt.ylabel("Эффективность (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig('efficiency_vs_threads.png')
    plt.show()

# График 7: Сравнение ускорения std::thread vs std::async
if not mt_data.empty and not async_data.empty:
    plt.figure(figsize=(12, 8))
    
    # Берем оптимальные параметры для каждого метода
    optimal_threads = {}
    optimal_async = {}
    
    for size in sorted(mt_data['matrixSize'].unique()):
        # Лучший результат для std::thread
        thread_size_data = mt_data[mt_data['matrixSize'] == size]
        if not thread_size_data.empty:
            best_thread = thread_size_data.loc[thread_size_data['time'].idxmin()]
            single_thread_time = thread_size_data[thread_size_data['threads'] == 1]['time']
            
            if len(single_thread_time) > 0:
                base_time = single_thread_time.values[0]
                optimal_threads[size] = {
                    'time': best_thread['time'],
                    'threads': best_thread['threads'],
                    'speedup': base_time / best_thread['time']
                }
        
        # Лучший результат для async
        async_size_data = async_data[async_data['matrixSize'] == size]
        if not async_size_data.empty:
            best_async = async_size_data.loc[async_size_data['time'].idxmin()]
            optimal_async[size] = {
                'time': best_async['time'],
                'speedup': base_time / best_async['time'] if len(single_thread_time) > 0 else 0
            }
    
    # Строим график сравнения
    if optimal_threads and optimal_async:
        sizes = sorted(optimal_threads.keys())
        thread_speedups = [optimal_threads[size]['speedup'] for size in sizes]
        async_speedups = [optimal_async[size]['speedup'] for size in sizes]
        thread_counts = [optimal_threads[size]['threads'] for size in sizes]
        
        plt.plot(sizes, thread_speedups, marker='o', linewidth=2, label='std::thread (оптимальные потоки)')
        plt.plot(sizes, async_speedups, marker='s', linewidth=2, label='std::async')
        
        # Добавляем аннотации с количеством потоков для std::thread
        for i, (size, threads) in enumerate(zip(sizes, thread_counts)):
            plt.annotate(f'{threads}t', (size, thread_speedups[i]), 
                        textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        
        plt.title("Сравнение ускорения: std::thread vs std::async")
        plt.xlabel("Размер матрицы")
        plt.ylabel("Ускорение (раз)")
        plt.legend()
        plt.grid(True)
        plt.savefig('thread_vs_async_speedup.png')
        plt.show()

# График 8: Время выполнения в зависимости от размера матрицы (3D поверхность)
try:
    from mpl_toolkits.mplot3d import Axes3D
    
    # Подготовка данных для 3D графика
    pivot_data = mt_data.pivot_table(values='time', index='matrixSize', columns='threads', aggfunc='min')
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    X = pivot_data.columns.values  # threads
    Y = pivot_data.index.values    # matrixSize
    X, Y = np.meshgrid(X, Y)
    Z = pivot_data.values
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('Количество потоков')
    ax.set_ylabel('Размер матрицы')
    ax.set_zlabel('Время (мс)')
    ax.set_title('3D: Время выполнения от размера матрицы и количества потоков')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig('3d_time_surface.png')
    plt.show()
    
except ImportError:
    print("3D графики недоступны (требуется mpl_toolkits)")

# Дополнительный анализ: таблица оптимальных параметров
print("\n=== ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ ДЛЯ МНОГОПОТОЧНОСТИ ===")
for size in sorted(mt_data['matrixSize'].unique()):
    size_data = mt_data[mt_data['matrixSize'] == size]
    if not size_data.empty:
        best_row = size_data.loc[size_data['time'].idxmin()]
        single_thread_time = size_data[size_data['threads'] == 1]['time']
        
        if len(single_thread_time) > 0:
            base_time = single_thread_time.values[0]
            speedup = base_time / best_row['time']
            efficiency = (speedup / best_row['threads']) * 100
            
            print(f"Размер {size}x{size}:")
            print(f"  Оптимальные параметры: {best_row['threads']} потоков, блок {best_row['blockSize']}")
            print(f"  Время: {best_row['time']:.0f} мс (ускорение: {speedup:.2f}x, эффективность: {efficiency:.1f}%)")
            
            # Сравнение с async
            async_best = async_data[(async_data['matrixSize'] == size) & 
                                  (async_data['blockSize'] == best_row['blockSize'])]
            if not async_best.empty:
                async_time = async_best['time'].min()
                async_speedup = base_time / async_time
                print(f"  Async: {async_time:.0f} мс (ускорение: {async_speedup:.2f}x)")
            print()