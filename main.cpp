#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <functional>
#include <fstream>
#include <iomanip>
#include <random>

#ifdef _WIN32
#include <windows.h>
#endif

#include "ThreadSafeList.hpp"
#include "CoarseGrainedList.hpp"

// Глобальный файл для записи результатов
std::ofstream resultsFile;
constexpr int OPERATIONS_PER_THREAD = 10000;
constexpr int NUM_RUNS = 3;  // Количество запусков для усреднения

// Конфигурации тестов: {читатели, писатели}
std::vector<std::pair<int, int>> configurations = {
    {1, 1}, {2, 1}, {4, 1}, {8, 1},
    {1, 2}, {1, 4}, {1, 8},
    {2, 2}, {4, 4}, {8, 8}
};

// Функция для измерения времени выполнения
double measureTime(std::function<void()> func) {
    auto start = std::chrono::steady_clock::now();
    func();
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    return elapsed.count();
}

// Запись результата в файл и консоль
void writeResult(const std::string& message) {
    std::cout << message;
    if (resultsFile.is_open()) {
        resultsFile << message;
        resultsFile.flush();
    }
}

// Тест корректности
void testCorrectness() {
    writeResult("=== Тест корректности CoarseGrainedList ===\n");
    
    CoarseGrainedList list;
    bool allTestsPassed = true;

    // Тест 1: Вставка элементов
    writeResult("1. Тест вставки элементов... ");
    bool insertOk = true;
    for (int i = 0; i < 10; i++) {
        std::string key = "key" + std::to_string(i);
        std::string value = "value" + std::to_string(i);
        if (!list.insert(key, value)) {
            insertOk = false;
            break;
        }
    }
    if (insertOk && list.size() == 10) {
        writeResult("УСПЕХ\n");
    } else {
        writeResult("ПРОВАЛ\n");
        allTestsPassed = false;
    }

    // Тест 2: Поиск элементов
    writeResult("2. Тест поиска элементов... ");
    bool findOk = true;
    std::string value;
    for (int i = 0; i < 10; i++) {
        std::string key = "key" + std::to_string(i);
        if (!list.find(key, value) || value != "value" + std::to_string(i)) {
            findOk = false;
            break;
        }
    }
    if (findOk) {
        writeResult("УСПЕХ\n");
    } else {
        writeResult("ПРОВАЛ\n");
        allTestsPassed = false;
    }

    // Тест 3: Удаление элементов
    writeResult("3. Тест удаления элементов... ");
    bool removeOk = true;
    for (int i = 0; i < 5; i++) {
        std::string key = "key" + std::to_string(i);
        if (!list.remove(key)) {
            removeOk = false;
            break;
        }
    }
    if (removeOk && list.size() == 5) {
        writeResult("УСПЕХ\n");
    } else {
        writeResult("ПРОВАЛ\n");
        allTestsPassed = false;
    }

    // Тест 4: Поиск после удаления
    writeResult("4. Тест поиска после удаления... ");
    bool findAfterRemoveOk = true;
    // Должны найти ключи 5-9
    for (int i = 5; i < 10; i++) {
        std::string key = "key" + std::to_string(i);
        if (!list.find(key, value)) {
            findAfterRemoveOk = false;
            break;
        }
    }
    // Не должны найти ключи 0-4
    for (int i = 0; i < 5; i++) {
        std::string key = "key" + std::to_string(i);
        if (list.find(key, value)) {
            findAfterRemoveOk = false;
            break;
        }
    }
    if (findAfterRemoveOk) {
        writeResult("УСПЕХ\n");
    } else {
        writeResult("ПРОВАЛ\n");
        allTestsPassed = false;
    }

    // Тест 5: Дубликаты
    writeResult("5. Тест дубликатов... ");
    if (!list.insert("key5", "new_value")) { // Должно вернуть false
        writeResult("УСПЕХ\n");
    } else {
        writeResult("ПРОВАЛ\n");
        allTestsPassed = false;
    }

    if (allTestsPassed) {
        writeResult("=== Все тесты пройдены УСПЕШНО ===\n");
    } else {
        writeResult("=== Некоторые тесты ПРОВАЛЕНЫ ===\n");
    }
    writeResult("\n");
}

// Функция для тестирования производительности
void benchmarkList(ThreadSafeList* list, int readerThreads, int writerThreads, int operationsPerThread) {
    std::vector<std::thread> threads;
    std::atomic<int> completed_operations{0};

    // Функция для читателя
    auto reader = [&](int id) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 99);
        
        for (int i = 0; i < operationsPerThread; ++i) {
            std::string key = "key" + std::to_string(dis(gen) % 50);
            std::string value;
            list->find(key, value);
            completed_operations++;
        }
    };

    // Функция для писателя
    auto writer = [&](int id) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, 999);
        
        for (int i = 0; i < operationsPerThread; ++i) {
            std::string key = "key" + std::to_string(dis(gen));
            std::string value = "val" + std::to_string(dis(gen));
            
            if (dis(gen) % 3 == 0) {
                // 33% удалений
                list->remove(key);
            } else {
                // 67% вставок
                list->insert(key, value);
            }
            completed_operations++;
        }
    };

    // Создаем потоки-читатели
    for (int i = 0; i < readerThreads; ++i) {
        threads.emplace_back(reader, i);
    }

    // Создаем потоки-писатели
    for (int i = 0; i < writerThreads; ++i) {
        threads.emplace_back(writer, i);
    }

    // Ожидаем завершения всех потоков
    for (auto &t : threads) {
        t.join();
    }
}

// Основной бенчмарк
void runPerformanceBenchmark() {
    writeResult("=== Performance Benchmark - CoarseGrainedList ===\n");

    // Заголовок таблицы для файла
    if (resultsFile.is_open()) {
        resultsFile << "Configuration,Readers,Writers,Time(s),Operations/sec\n";
    }

    for (const auto &config : configurations) {
        int readers = config.first;
        int writers = config.second;

        std::string configHeader = "\n--- Configuration: " + std::to_string(readers) +
                                   " readers, " + std::to_string(writers) + " writers ---\n";
        writeResult(configHeader);

        double totalTime = 0;
        double totalOpsPerSec = 0;

        for (int run = 0; run < NUM_RUNS; ++run) {
            CoarseGrainedList list;
            
            // Предварительное заполнение
            for (int i = 0; i < 100; i++) {
                list.insert("pre_key" + std::to_string(i), "pre_val" + std::to_string(i));
            }

            double time = measureTime([&]() {
                benchmarkList(&list, readers, writers, OPERATIONS_PER_THREAD);
            });

            totalTime += time;
            
            int totalOperations = (readers + writers) * OPERATIONS_PER_THREAD;
            double opsPerSec = totalOperations / time;
            totalOpsPerSec += opsPerSec;

            std::string result = "Run " + std::to_string(run + 1) + ": " + 
                                std::to_string(time) + "s, " +
                                std::to_string(opsPerSec) + " ops/sec\n";
            writeResult(result);
        }

        double avgTime = totalTime / NUM_RUNS;
        double avgOpsPerSec = totalOpsPerSec / NUM_RUNS;

        std::string avgResult = "СРЕДНЕЕ: " + 
                               std::to_string(avgTime) + "s, " +
                               std::to_string(avgOpsPerSec) + " ops/sec\n";
        writeResult(avgResult);

        // Записываем данные в CSV формат
        if (resultsFile.is_open()) {
            resultsFile << readers << "R_" << writers << "W,"
                        << readers << "," << writers << ","
                        << std::fixed << std::setprecision(6) << avgTime << ","
                        << avgOpsPerSec << "\n";
        }
    }
}

// Простой тест для демонстрации работы
void simpleTest() {
    std::cout << "\n=== Простой тест функциональности ===" << std::endl;

    CoarseGrainedList list;

    // Вставка элементов
    std::cout << "1. Добавление элементов..." << std::endl;
    for (int i = 1; i <= 5; i++) {
        std::string key = "key" + std::to_string(i);
        std::string value = "value" + std::to_string(i);
        if (list.insert(key, value)) {
            std::cout << "   Добавлен: " << key << " -> " << value << std::endl;
        }
    }
    list.print();
    std::cout << "Размер: " << list.size() << std::endl;

    // Поиск элементов
    std::cout << "\n2. Поиск элементов..." << std::endl;
    std::string value;
    if (list.find("key3", value)) {
        std::cout << "   Найден key3: " << value << std::endl;
    }
    if (!list.find("key10", value)) {
        std::cout << "   key10 не найден (корректно)" << std::endl;
    }

    // Удаление элементов
    std::cout << "\n3. Удаление элементов..." << std::endl;
    if (list.remove("key2")) {
        std::cout << "   Удален key2" << std::endl;
    }
    if (list.remove("key4")) {
        std::cout << "   Удален key4" << std::endl;
    }
    list.print();
    std::cout << "Размер после удаления: " << list.size() << std::endl;

    // Попытка добавить дубликат
    std::cout << "\n4. Попытка добавить дубликат..." << std::endl;
    if (!list.insert("key3", "new_value")) {
        std::cout << "   Дубликат key3 не добавлен (корректно)" << std::endl;
    }

    std::cout << "\n5. Финальное состояние:" << std::endl;
    list.print();
    std::cout << "Пустой список: " << (list.isEmpty() ? "да" : "нет") << std::endl;
    std::cout << "Размер: " << list.size() << std::endl;
}

// Функция для создания отчета в формате CSV
void createCSVReport() {
    std::ofstream csvFile("coarse_grained_performance.csv");
    if (!csvFile.is_open()) {
        std::cerr << "Не удалось создать файл отчета CSV" << std::endl;
        return;
    }

    writeResult("\n=== Создание CSV отчета ===\n");

    // Заголовок CSV
    csvFile << "Configuration,Readers,Writers,Time(s),Operations/sec\n";
    
    for (const auto& config : configurations) {
        int readers = config.first;
        int writers = config.second;

        double totalTime = 0;
        double totalOpsPerSec = 0;

        for (int run = 0; run < NUM_RUNS; ++run) {
            CoarseGrainedList list;
            
            // Предварительное заполнение
            for (int i = 0; i < 100; i++) {
                list.insert("pre_key" + std::to_string(i), "pre_val" + std::to_string(i));
            }

            double time = measureTime([&]() {
                benchmarkList(&list, readers, writers, OPERATIONS_PER_THREAD);
            });

            totalTime += time;
            totalOpsPerSec += (readers + writers) * OPERATIONS_PER_THREAD / time;
        }

        double avgTime = totalTime / NUM_RUNS;
        double avgOpsPerSec = totalOpsPerSec / NUM_RUNS;

        // Запись в CSV
        csvFile << readers << "R_" << writers << "W,"
                << readers << "," << writers << ","
                << std::fixed << std::setprecision(6) << avgTime << ","
                << avgOpsPerSec << "\n";

        std::cout << "Записана конфигурация: " << readers << " читателей, " 
                  << writers << " писателей" << std::endl;
    }

    csvFile.close();
    writeResult("CSV отчет создан: coarse_grained_performance.csv\n");
}

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    // Открываем файл для записи результатов
    resultsFile.open("benchmark_results.txt");
    if (!resultsFile.is_open()) {
        std::cerr << "Не удалось открыть файл для записи результатов" << std::endl;
        return 1;
    }

    try {
        writeResult("=== Тестирование CoarseGrainedList (Грубая блокировка) ===\n\n");
        
        // Запускаем тесты
        testCorrectness();
        simpleTest();
        runPerformanceBenchmark();

        // Создаем дополнительный CSV отчет
        createCSVReport();

        writeResult("\n=== Тестирование завершено ===\n");

    } catch (const std::exception &e) {
        writeResult("Ошибка: " + std::string(e.what()) + "\n");
        return 1;
    }

    resultsFile.close();
    return 0;
}