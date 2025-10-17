#ifndef COARSE_GRAINED_LIST_HPP
#define COARSE_GRAINED_LIST_HPP

#include "ThreadSafeList.hpp"
#include <mutex>

class CoarseGrainedList : public ThreadSafeList {
private:
    Node* head;
    mutable std::mutex mtx;
    std::atomic<size_t> list_size{0};
    
public:
    CoarseGrainedList() : head(nullptr) {}
    
    ~CoarseGrainedList() override {
        clear();
    }
    
    bool insert(const std::string& key, const std::string& value) override {
        if (key.empty()) return false;
        
        std::lock_guard<std::mutex> lock(mtx);
        
        // Создаем новый узел
        Node* new_node = new Node(key, value);
        
        // Если список пустой
        if (!head) {
            head = new_node;
            list_size++;
            return true;
        }
        
        // Если вставляем в начало (для поддержания порядка)
        if (head->key > key) {
            new_node->next = head;
            head = new_node;
            list_size++;
            return true;
        }
        
        // Если ключ уже существует в голове
        if (head->key == key) {
            delete new_node;
            return false;
        }
        
        // Ищем позицию для вставки
        Node* current = head;
        while (current->next) {
            if (current->next->key == key) {
                delete new_node;
                return false; // Ключ уже существует
            }
            if (current->next->key > key) {
                break; // Нашли позицию
            }
            current = current->next;
        }
        
        // Вставляем новый узел
        new_node->next = current->next;
        current->next = new_node;
        list_size++;
        return true;
    }
    
    bool remove(const std::string& key) override {
        if (key.empty()) return false;
        
        std::lock_guard<std::mutex> lock(mtx);
        
        if (!head) return false;
        
        // Удаление из головы
        if (head->key == key) {
            Node* to_delete = head;
            head = head->next;
            delete to_delete;
            list_size--;
            return true;
        }
        
        // Ищем узел для удаления
        Node* current = head;
        while (current->next) {
            if (current->next->key == key) {
                Node* to_delete = current->next;
                current->next = to_delete->next;
                delete to_delete;
                list_size--;
                return true;
            }
            if (current->next->key > key) {
                return false; // Ключ не найден
            }
            current = current->next;
        }
        
        return false;
    }
    
    bool find(const std::string& key, std::string& value) override {
        if (key.empty()) return false;
        
        std::lock_guard<std::mutex> lock(mtx);
        
        Node* current = head;
        while (current) {
            if (current->key == key) {
                value = current->value;
                return true;
            }
            if (current->key > key) {
                return false; // Ключ не найден (список отсортирован)
            }
            current = current->next;
        }
        
        return false;
    }
    
    size_t size() const override {
        return list_size.load();
    }
    
    bool isEmpty() const override {
        return list_size.load() == 0;
    }
    
    void print() override {
        std::lock_guard<std::mutex> lock(mtx);
        
        Node* current = head;
        std::cout << "Список: ";
        while (current) {
            std::cout << "[" << current->key << ":" << current->value << "]";
            current = current->next;
            if (current) std::cout << " -> ";
        }
        std::cout << " -> nullptr" << std::endl;
    }

private:
    void clear() {
        std::lock_guard<std::mutex> lock(mtx);
        Node* current = head;
        while (current) {
            Node* next = current->next;
            delete current;
            current = next;
        }
        head = nullptr;
        list_size = 0;
    }
};

#endif