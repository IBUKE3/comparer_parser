from app_back import (
    file_2_tokens,
    tokens_counter_2_lemms,
    file_2_ners,
    compare_2_files,
    calc_cos_sim,
    ENCODING
)

def print_compared(file1, file2: str):
    res = compare_2_files(file1, file2)
    print("________________________________________________________________________")
    print("Результаты сравнения двух файлов:")
    print("Пересечение множества лемм:", res["intersection"])
    print("Множество лемм первого файла без лемм из второго:", res["l1_without_l2"])
    print("Множество лемм второго файла без лемм из первого:", res["l2_without_l1"])
    print("Коэффициент Жаккара:", res["jaccar"])
    print("Косинусная близость TF-IDF векторов на основе лемм:", res["cos_sim"])
    print("Покрытие словарей друг другом:", res["intersection"])
    print("________________________________________________________________________")


v1 = [1, 1, 1]
v2 = [-1, -1, -1]
v3 = [0, 1, -1]

for v in (v1, v2, v3):
    print(f"Косинусная близость {v1} и {v}: {calc_cos_sim(v1, v)}")

fn1 = input("Введите путь к первому файлу: ")
fn2 = input("Введите путь ко второму файлу: ")

print("Токены первого файла")
t1 = file_2_tokens(fn1)
print(t1)
print("Токены второго файла")
t2 = file_2_tokens(fn2)
print(t2)
print("_________________________________")
print("Леммы первого файла")
print(tokens_counter_2_lemms(t1))
print("Леммы второго файла")
print(tokens_counter_2_lemms(t2))
print("_________________________________")
print("Именованные сущности первого файла")
print(file_2_ners(fn1))
print("Именованные сущности первого файла")
print(file_2_ners(fn2))
print_compared(fn1, fn2)
