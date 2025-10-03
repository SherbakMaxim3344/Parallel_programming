#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <algorithm>
#include <future>
#include <cmath>
#include <chrono>
#include <fstream>
#include <string>
#include <functional>

class Matrix{
private:
    std::vector<std::vector<int>> data; // Двумерный вектор для хранения данных
    size_t n;

public:
    size_t getSize() const{
        return n;
    }

    // *Конструктор: создает квадратную матрицу n x n со случайными значениями
    Matrix(size_t n, int minVal, int maxVal) : n(n){
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(minVal, maxVal);

        // Заполняем матрицу случайными числами
        data.resize(n, std::vector<int>(n));
        for (size_t i = 0; i < n; ++i){
            for (size_t j = 0; j < n; ++j){
                data[i][j] = dist(gen);
            }
        }
    }

    // *Конструктор нулевой матрицы
    Matrix(size_t n, int value) : n(n){
        data.resize(n, std::vector<int>(n, value));
    }

    // *Метод для вывода матрицы
    void print() const{
        for (size_t i = 0; i < n; ++i){
            for (size_t j = 0; j < n; ++j){
                std::cout << data[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // *Оператор доступа к элементам через [][]
    std::vector<int> &operator[](size_t index) { return data[index]; }
    const std::vector<int> &operator[](size_t index) const { return data[index]; }
    int &operator()(size_t i, size_t j) { return data[i][j]; }
    const int &operator()(size_t i, size_t j) const { return data[i][j]; }
};

// ? однопоточное умножение
Matrix multiplyBase(const Matrix &A, const Matrix &B){
    int N = A.getSize();
    Matrix res(N, 0);

    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            for (int k = 0; k < N; k++){
                res(i, j) += A(i, k) * B(k, j);
            }
        }
    }
    return res;
}
// *Структура для хранения информации о блоке
struct BlockTask{
    int i, j; // Начальные координаты блока
};
// * Функция вычисления одного блока
std::vector<std::vector<int>> computeBlock(
                const Matrix &A, const Matrix &B,
                int startI, int startJ, int blockSize){
    int n=A.getSize();
    int blockRows=std::min(blockSize, n -startI);
    int blockCols=std::min(blockSize, n-startJ);

    std::vector<std::vector<int>> block(blockRows, std::vector<int>(blockCols,0));

    for(int i=0; i<blockRows; ++i){
        for(int j=0; j<blockCols; ++j){
            for(int k=0; k<n; ++k){
                block[i][j]+=A[startI+i][k]*B[k][startJ+j];
            }
        }
    }
    return block;
}
// ? блочное умножение с thread
Matrix multiplyBlockedThreads(
    const Matrix &A, const Matrix &B,
    int blockSize, int numThreads = 0){
    int n = A.getSize();
    Matrix res(n, 0);

    // Создаем задачи для всех блоков
    std::vector<BlockTask> tasks;
    for (int i = 0; i < n; i += blockSize){
        for (int j = 0; j < n; j += blockSize){
            tasks.push_back(BlockTask{i, j});
        }
    }
    // Автоматическое определение количества потоков
    if (numThreads == 0){
        numThreads = std::thread::hardware_concurrency(); // определяет наилучшее кол-во потоков для пк
    }

    std::vector<std::thread> threads;

    // Функция-воркер для каждого потока
    auto worker = [&](int threadId){
        for (size_t t = threadId; t < tasks.size(); t += numThreads){
            auto block = computeBlock(A, B, tasks[t].i, tasks[t].j, blockSize);

            // Копируем вычисленный блок в результирующую матрицу
            int blockRows=block.size();
            int blockCols=block[0].size();

            for(int bi=0;bi<blockRows;++bi){
                for(int bj=0; bj<blockCols; ++bj){
                    res(tasks[t].i+bi,tasks[t].j+bj)=block[bi][bj];
                }
            }
        }
    };

    // Запускаем потоки
    for(int t=0; t<numThreads; ++t){
        threads.emplace_back(worker, t);
    }
    
    // Ждем завершения всех потоков
    for (auto &th : threads) {
        th.join();
    }
    
    return res;
}

// ? блочное умножение с async/future
// * Функция для запуска задачи блока (нужна для std::async)
std::pair<BlockTask, std::vector<std::vector<int>>> runBlockTask(
    const Matrix &A, const Matrix &B, BlockTask task, int blockSize) {
    auto block = computeBlock(A, B, task.i, task.j, blockSize);
    return {task, block};
}

Matrix multiplyBlockedAsync(
            const Matrix &A, const Matrix &B, int blockSize) {
    int n = A.getSize();
    Matrix result(n, 0);
    
    std::vector<std::future<std::pair<BlockTask, std::vector<std::vector<int>>>>> futures;
    for (int i = 0; i < n; i += blockSize) {
        for (int j = 0; j < n; j += blockSize) {
            BlockTask task{i, j};
            futures.push_back(
                std::async(std::launch::async, runBlockTask, 
                std::cref(A), std::cref(B), task, blockSize)
            );
        }
    }
        // Собираем результаты
    for (auto &f : futures) {
        auto [task, block] = f.get();  // Получаем результат
        int rows = block.size();
        int cols = block[0].size();
        
        // Копируем вычисленный блок в результирующую матрицу
        for (int bi = 0; bi < rows; ++bi) {
            for (int bj = 0; bj < cols; ++bj) {
                result(task.i + bi, task.j + bj) = block[bi][bj];
            }
        }
    }
    
    return result;
}


// * Функция для проверки точности между двумя матрицами
bool checkEqual(const Matrix &base, const Matrix &tested, double epsilon = 1e-10) {
    int n = base.getSize();
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double diff = std::fabs(base(i, j) - tested(i, j));
            if (diff > epsilon) {
                std::cout << "Различие найдено в элементе [" << i << "][" << j << "]: " 
                        << base(i, j) << " vs " << tested(i, j) 
                        << " (разница: " << diff << ")" << std::endl;
                return false;
            }
        }
    }
    return true;
}

// * Функция для измерения времени выполнения
template<typename Func>
long long measureTime(Func func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

// * Инициализация файла результатов
void initResultsFile(const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << "Тип,Размер матрицы,Размер блока,Количество потоков,Время (мс)\n";
        file.close();
    }
}

// * Сохранение результатов в CSV
void saveResults(const std::string& filename, const std::string& type, 
                int matrixSize, int blockSize, int numThreads, long long time) {
    std::ofstream file(filename, std::ios::app);
    if (file.is_open()) {
        file << type << "," << matrixSize << "," << blockSize << "," 
            << numThreads << "," << time << "\n";
        file.close();
    }
}

// * Функция бенчмарка
void benchmark() {
    std::string filename = "benchmark_results.csv";
    initResultsFile(filename);
    
    std::vector<int> numThreads = {1, 2, 4, 8, 12, 16, 32};
    std::vector<int> matrixSizes = {256, 512, 1024, 2048};
    std::vector<int> blockSizes = {16, 32, 64, 128};
    
    std::vector<std::string> types = {
        "Многопоточное умножение", 
        "Async умножение", 
        "Стандартное умножение"
    };

    for (auto n : matrixSizes) {
        std::cout << "\n=== Тестирование матриц " << n << "x" << n << " ===" << std::endl;
        
        // Генерация тестовых матриц
        Matrix A(n, 0, 9);
        Matrix B(n, 0, 9);
        
        Matrix CStandard(n, 0);

        for (auto blockSize : blockSizes) {
            std::cout << "Размер блока: " << blockSize << std::endl;
            
            // Стандартное умножение (эталон)
            if (n <= 2048) { // Для больших матриц стандартное умножение может быть очень медленным
                long long timeStandard = measureTime([&]() {
                    CStandard = multiplyBase(A, B);
                });
                saveResults(filename, types[2], n, blockSize, 1, timeStandard);
                std::cout << "  Стандартное: " << timeStandard << " мс" << std::endl;
            }

            // Многопоточное умножение с разным количеством потоков
            for (auto threads : numThreads) {
                Matrix CThread(n, 0);
                long long timeThreads = measureTime([&]() {
                    CThread = multiplyBlockedThreads(A, B, blockSize, threads);
                });

                // Проверка корректности
                if (n <= 1024) {
                    if (!checkEqual(CStandard, CThread)) {
                        std::cerr << "ОШИБКА: Threaded результат не совпадает со стандартным!" << std::endl;
                    }
                }

                saveResults(filename, types[0], n, blockSize, threads, timeThreads);
                std::cout << "  Потоков " << threads << ": " << timeThreads << " мс" << std::endl;
            }

            // Async умножение
            Matrix CAsync(n, 0);
            long long timeAsync = measureTime([&]() {
                CAsync = multiplyBlockedAsync(A, B, blockSize);
            });

            // Проверка корректности
            if (n <= 1024) {
                if (!checkEqual(CStandard, CAsync)) {
                    std::cerr << "ОШИБКА: Async результат не совпадает со стандартным!" << std::endl;
                }
            }

            saveResults(filename, types[1], n, blockSize, 0, timeAsync);
            std::cout << "  Async: " << timeAsync << " мс" << std::endl;
        }
    }
    
    std::cout << "\nБенчмарк завершен! Результаты сохранены в " << filename << std::endl;
}

// * Быстрый тест на корректность
void quickTest() {
    std::cout << "=== БЫСТРЫЙ ТЕСТ КОРРЕКТНОСТИ ===" << std::endl;
    
    Matrix A(4, 0, 9);
    Matrix B(4, 0, 9);
    
    std::cout << "Матрица A:" << std::endl;
    A.print();
    std::cout << "Матрица B:" << std::endl;
    B.print();
    
    Matrix standard = multiplyBase(A, B);
    Matrix threaded = multiplyBlockedThreads(A, B, 2, 2);
    Matrix async = multiplyBlockedAsync(A, B, 2);
    
    std::cout << "Стандартное умножение:" << std::endl;
    standard.print();
    
    std::cout << "Многопоточное умножение:" << std::endl;
    threaded.print();
    
    std::cout << "Async умножение:" << std::endl;
    async.print();
    
    bool correct1 = checkEqual(standard, threaded);
    bool correct2 = checkEqual(standard, async);
    
    if (correct1 && correct2) {
        std::cout << "✓ Все методы дают идентичные результаты!" << std::endl;
    } else {
        std::cout << "✗ Обнаружены ошибки в реализации!" << std::endl;
    }
}

int main() {
    // Быстрый тест корректности
    quickTest();
    
    // Полный бенчмарк
    std::cout << "\nЗапустить полный бенчмарк? (y/n): ";
    char choice;
    std::cin >> choice;
    
    if (choice == 'y' || choice == 'Y') {
        benchmark();
    }
    
    return 0;
}