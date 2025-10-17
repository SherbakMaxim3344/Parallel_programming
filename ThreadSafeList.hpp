#ifndef THREAD_SAFE_LIST_HPP
#define THREAD_SAFE_LIST_HPP

#include <iostream>
#include <string>
#include <mutex>
#include <atomic>

// Базовый интерфейс для всех реализаций
class ThreadSafeList {
public:
    virtual ~ThreadSafeList() = default;
    virtual bool insert(const std::string& key, const std::string& value) = 0;
    virtual bool remove(const std::string& key) = 0;
    virtual bool find(const std::string& key, std::string& value) = 0;
    virtual size_t size() const = 0;
    virtual void print() = 0;
    virtual bool isEmpty() const = 0;
};

// Узел списка
struct Node {
    std::string key;
    std::string value;
    Node* next;
    
    Node(const std::string& k, const std::string& v) 
        : key(k), value(v), next(nullptr) {}
};

#endif