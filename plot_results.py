import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Создаем папку для графиков если её нет
os.makedirs('graphs', exist_ok=True)

def read_and_prepare_data(filename):
    """Чтение и подготовка данных из CSV файла"""
    df = pd.read_csv(filename)
    
    # Извлекаем количество читателей и писателей из колонки Configuration
    df[['Readers', 'Writers']] = df['Configuration'].str.extract(r'(\d+)R_(\d+)W')
    df['Readers'] = df['Readers'].astype(int)
    df['Writers'] = df['Writers'].astype(int)
    
    # Добавляем колонку с соотношением читателей/писателей
    df['Reader_Writer_Ratio'] = df['Readers'] / df['Writers']
    df['Total_Threads'] = df['Readers'] + df['Writers']
    
    return df

def plot_performance_by_threads(df):
    """График производительности в зависимости от общего количества потоков"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # График 1: Время выполнения vs Общее количество потоков
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    for i, row in df.iterrows():
        ax1.scatter(row['Total_Threads'], row['Time(s)'], 
                   color=colors[i], s=100, alpha=0.7)
        ax1.annotate(f"{row['Readers']}R/{row['Writers']}W", 
                    (row['Total_Threads'], row['Time(s)']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('Общее количество потоков')
    ax1.set_ylabel('Время выполнения (секунды)')
    ax1.set_title('Время выполнения vs Количество потоков\n(CoarseGrainedList)')
    ax1.grid(True, alpha=0.3)
    
    # График 2: Операций в секунду vs Общее количество потоков
    for i, row in df.iterrows():
        ax2.scatter(row['Total_Threads'], row['Operations/sec'], 
                   color=colors[i], s=100, alpha=0.7)
        ax2.annotate(f"{row['Readers']}R/{row['Writers']}W", 
                    (row['Total_Threads'], row['Operations/sec']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_xlabel('Общее количество потоков')
    ax2.set_ylabel('Операций в секунду')
    ax2.set_title('Производительность vs Количество потоков\n(CoarseGrainedList)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('graphs/performance_by_threads.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_reader_writer_analysis(df):
    """Анализ влияния соотношения читателей/писателей"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Группируем по соотношению читателей/писателей
    ratio_groups = df.groupby('Reader_Writer_Ratio')
    
    # График 1: Средняя производительность по соотношениям
    ratios = []
    avg_ops = []
    std_ops = []
    
    for ratio, group in ratio_groups:
        ratios.append(ratio)
        avg_ops.append(group['Operations/sec'].mean())
        std_ops.append(group['Operations/sec'].std())
    
    bars = ax1.bar(range(len(ratios)), avg_ops, yerr=std_ops, 
                   capsize=5, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Соотношение читателей/писателей')
    ax1.set_ylabel('Операций в секунду')
    ax1.set_title('Производительность по соотношению читателей/писателей\n(CoarseGrainedList)')
    ax1.set_xticks(range(len(ratios)))
    ax1.set_xticklabels([f'{r:.1f}' for r in ratios], rotation=45)
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars, avg_ops):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000,
                f'{value/1000:.0f}K', ha='center', va='bottom', fontsize=9)
    
    # График 2: Тепловая карта производительности
    pivot_df = df.pivot_table(values='Operations/sec', 
                             index='Readers', 
                             columns='Writers', 
                             aggfunc='mean')
    
    im = ax2.imshow(pivot_df.values, cmap='YlOrRd', aspect='auto')
    ax2.set_xlabel('Писатели')
    ax2.set_ylabel('Читатели')
    ax2.set_title('Тепловая карта производительности\n(CoarseGrainedList)')
    
    # Добавляем подписи
    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            text = ax2.text(j, i, f'{pivot_df.iloc[i, j]/1000:.0f}K',
                           ha="center", va="center", color="black", fontsize=8)
    
    ax2.set_xticks(range(len(pivot_df.columns)))
    ax2.set_xticklabels(pivot_df.columns)
    ax2.set_yticks(range(len(pivot_df.index)))
    ax2.set_yticklabels(pivot_df.index)
    
    plt.colorbar(im, ax=ax2, label='Операций в секунду')
    plt.tight_layout()
    plt.savefig('graphs/reader_writer_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_comparison_analysis(df):
    """Сравнительный анализ разных конфигураций"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Сортируем по производительности
    df_sorted = df.sort_values('Operations/sec', ascending=False)
    
    # График 1: Сравнение производительности всех конфигураций
    bars = ax1.bar(range(len(df_sorted)), df_sorted['Operations/sec'], 
                   color=plt.cm.plasma(np.linspace(0, 1, len(df_sorted))))
    ax1.set_xlabel('Конфигурация потоков')
    ax1.set_ylabel('Операций в секунду')
    ax1.set_title('Сравнение производительности всех конфигураций\n(CoarseGrainedList)')
    ax1.set_xticks(range(len(df_sorted)))
    ax1.set_xticklabels([f"{row['Readers']}R/{row['Writers']}W" 
                        for _, row in df_sorted.iterrows()], rotation=45)
    
    # Добавляем значения на столбцы
    for i, (bar, value) in enumerate(zip(bars, df_sorted['Operations/sec'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5000,
                f'{value/1000:.0f}K', ha='center', va='bottom', fontsize=9)
    
    # График 2: Время выполнения по конфигурациям
    bars2 = ax2.bar(range(len(df_sorted)), df_sorted['Time(s)'], 
                   color=plt.cm.viridis(np.linspace(0, 1, len(df_sorted))))
    ax2.set_xlabel('Конфигурация потоков')
    ax2.set_ylabel('Время выполнения (секунды)')
    ax2.set_title('Время выполнения по конфигурациям\n(CoarseGrainedList)')
    ax2.set_xticks(range(len(df_sorted)))
    ax2.set_xticklabels([f"{row['Readers']}R/{row['Writers']}W" 
                        for _, row in df_sorted.iterrows()], rotation=45)
    
    # Добавляем значения на столбцы
    for i, (bar, value) in enumerate(zip(bars2, df_sorted['Time(s)'])):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{value:.3f}s', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('graphs/comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_trend_analysis(df):
    """Анализ трендов производительности"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # График 1: Тренд при увеличении читателей (при фиксированных писателях)
    writer_configs = df['Writers'].unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(writer_configs)))
    
    for writer, color in zip(writer_configs, colors):
        subset = df[df['Writers'] == writer].sort_values('Readers')
        if len(subset) > 1:
            ax1.plot(subset['Readers'], subset['Operations/sec'], 
                    marker='o', linewidth=2, markersize=8, color=color,
                    label=f'{writer} писатель(ей)')
    
    ax1.set_xlabel('Количество читателей')
    ax1.set_ylabel('Операций в секунду')
    ax1.set_title('Влияние количества читателей на производительность\n(CoarseGrainedList)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График 2: Тренд при увеличении писателей (при фиксированных читателях)
    reader_configs = df['Readers'].unique()
    colors = plt.cm.Set2(np.linspace(0, 1, len(reader_configs)))
    
    for reader, color in zip(reader_configs, colors):
        subset = df[df['Readers'] == reader].sort_values('Writers')
        if len(subset) > 1:
            ax2.plot(subset['Writers'], subset['Operations/sec'], 
                    marker='s', linewidth=2, markersize=8, color=color,
                    label=f'{reader} читатель(ей)')
    
    ax2.set_xlabel('Количество писателей')
    ax2.set_ylabel('Операций в секунду')
    ax2.set_title('Влияние количества писателей на производительность\n(CoarseGrainedList)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('graphs/trend_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_report(df):
    """Создание сводного отчета"""
    print("=" * 60)
    print("СВОДНЫЙ ОТЧЕТ ПРОИЗВОДИТЕЛЬНОСТИ CoarseGrainedList")
    print("=" * 60)
    
    # Лучшая конфигурация
    best_config = df.loc[df['Operations/sec'].idxmax()]
    print(f"Лучшая конфигурация: {best_config['Readers']}R/{best_config['Writers']}W")
    print(f"Максимальная производительность: {best_config['Operations/sec']:,.0f} оп/сек")
    print(f"Время выполнения: {best_config['Time(s)']:.3f} сек")
    
    print("\n" + "-" * 60)
    
    # Анализ по соотношениям
    print("Производительность по соотношениям читателей/писателей:")
    ratio_stats = df.groupby('Reader_Writer_Ratio')['Operations/sec'].agg(['mean', 'std'])
    for ratio, stats in ratio_stats.iterrows():
        print(f"  Соотношение {ratio:.1f}: {stats['mean']:,.0f} ± {stats['std']:,.0f} оп/сек")
    
    print("\n" + "-" * 60)
    
    # Топ-5 конфигураций
    print("Топ-5 конфигураций по производительности:")
    top5 = df.nlargest(5, 'Operations/sec')
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"  {i}. {row['Readers']}R/{row['Writers']}W: {row['Operations/sec']:,.0f} оп/сек")
    
    print("=" * 60)

def main():
    """Основная функция"""
    csv_file = 'coarse_grained_performance.csv'
    
    if not os.path.exists(csv_file):
        print(f"Ошибка: Файл {csv_file} не найден!")
        print("Сначала запустите C++ бенчмарк для генерации данных")
        return
    
    print("Чтение данных из CSV файла...")
    df = read_and_prepare_data(csv_file)
    
    print(f"Загружено {len(df)} конфигураций тестирования")
    print("\nСоздание графиков...")
    
    # Создаем все графики
    plot_performance_by_threads(df)
    plot_reader_writer_analysis(df)
    plot_comparison_analysis(df)
    plot_trend_analysis(df)
    
    # Создаем сводный отчет
    create_summary_report(df)
    
    print(f"\nВсе графики сохранены в папку 'graphs/'")
    print("Доступные графики:")
    print("  - performance_by_threads.png: Производительность vs количество потоков")
    print("  - reader_writer_analysis.png: Анализ соотношения читателей/писателей")
    print("  - comparison_analysis.png: Сравнение всех конфигураций")
    print("  - trend_analysis.png: Анализ трендов производительности")

if __name__ == "__main__":
    main()