import json
import csv
from itertools import combinations
from io import StringIO
import sys
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel


def is_unique_combination(entities, attributes):
    """
    Проверяет, является ли комбинация атрибутов уникальной для всех сущностей.

    :param entities: список сущностей
    :param attributes: комбинация атрибутов
    :return: True, если комбинация уникальна, иначе False
    """
    seen = set()
    for entity in entities:
        identifier = tuple(entity.get(attr, None) for attr in attributes)
        if identifier in seen:
            return False
        seen.add(identifier)
    return True


def minimal_cover_algorithm(entities, important_attributes):
    """
    Реализует алгоритм минимального покрытия для нахождения минимального уникального набора атрибутов.

    :param entities: список сущностей
    :param important_attributes: список важных атрибутов
    :return: минимальная уникальная комбинация атрибутов
    """
    if not entities:
        return []

    unique_combinations = []

    for r in range(1, len(important_attributes) + 1):
        for combination in combinations(important_attributes, r):
            if is_unique_combination(entities, combination):
                unique_combinations.append(combination)

    # Проверяем, найден ли хоть один уникальный набор атрибутов
    if not unique_combinations:
        raise ValueError("No unique combination found")

    # Находим минимальное покрытие
    minimal_combination = min(unique_combinations, key=len)

    return list(minimal_combination)


def select_important_attributes(entities):
    """
    Отбирает важные атрибуты с помощью решающего дерева и pandas.get_dummies.

    :param entities: список сущностей
    :return: список важных атрибутов
    """
    df = pd.DataFrame(entities)
    X = df.drop(columns=["target"])
    y = df["target"]

    # Преобразование категориальных данных в бинарные признаки
    X_encoded = pd.get_dummies(X)

    clf = DecisionTreeClassifier()
    clf.fit(X_encoded, y)

    model = SelectFromModel(clf, prefit=True)
    important_features = model.get_support(indices=True)
    important_attributes = list(X_encoded.columns[important_features])

    # Возвращаем оригинальные имена атрибутов
    important_attributes = list(set([col.split('_')[0] for col in important_attributes]))

    return important_attributes


def main(json_data):
    """
    Основная функция, которая принимает JSON-строку и возвращает CSV-строку.

    :param json_data: JSON-строка с исходными данными
    :return: CSV-строка с минимальной уникальной комбинацией атрибутов
    """
    entities = json.loads(json_data)

    # Добавляем искусственный атрибут "target" для использования в классификации
    for i, entity in enumerate(entities):
        entity["target"] = i

    important_attributes = select_important_attributes(entities)
    minimal_combination = minimal_cover_algorithm(entities, important_attributes)

    output = StringIO()
    writer = csv.writer(output, lineterminator='\n')
    for attribute in minimal_combination:
        writer.writerow([attribute])

    return output.getvalue()


# Unit-тесты
import unittest


class TestUniqueCombination(unittest.TestCase):

    def setUp(self):
        self.json_data_1 = json.dumps([
            {"фамилия": "Смирнов", "имя": "Евгений", "отчество": "Александрович", "класс": "6"},
            {"фамилия": "Смирнов", "имя": "Евгений", "отчество": "Александрович", "класс": "7"},
            {"фамилия": "Петров", "имя": "Иван", "отчество": "Сергеевич", "класс": "7"}
        ])
        self.json_data_2 = json.dumps([
            {"фамилия": "Иванов", "имя": "Сергей", "отчество": "Петрович", "класс": "10", "школа": "1"},
            {"фамилия": "Сидоров", "имя": "Алексей", "отчество": "Иванович", "класс": "10", "школа": "1"},
            {"фамилия": "Иванов", "имя": "Алексей", "отчество": "Сергеевич", "класс": "10", "школа": "2"},
            {"фамилия": "Петров", "имя": "Иван", "отчество": "Алексеевич", "класс": "11", "школа": "1"}
        ])
        self.json_data_3 = json.dumps([
            {"фамилия": "Смирнов", "имя": "Евгений", "отчество": "Александрович", "класс": "6", "школа": "1"},
            {"фамилия": "Смирнов", "имя": "Евгений", "отчество": "Александрович", "класс": "6", "школа": "2"},
            {"фамилия": "Иванов", "имя": "Алексей", "отчество": "Сергеевич", "класс": "6", "школа": "1"},
            {"фамилия": "Петров", "имя": "Иван", "отчество": "Алексеевич", "класс": "6", "школа": "2"},
            {"фамилия": "Сидоров", "имя": "Алексей", "отчество": "Иванович", "класс": "6", "школа": "1"}
        ])
        self.entities_1 = json.loads(self.json_data_1)
        self.entities_2 = json.loads(self.json_data_2)
        self.entities_3 = json.loads(self.json_data_3)

    def test_is_unique_combination(self):
        self.assertTrue(is_unique_combination(self.entities_1, ["фамилия", "имя", "отчество", "класс"]))
        self.assertFalse(is_unique_combination(self.entities_1, ["фамилия", "имя", "отчество"]))
        self.assertTrue(is_unique_combination(self.entities_2, ["фамилия", "имя", "класс", "школа"]))
        self.assertFalse(is_unique_combination(self.entities_2, ["фамилия", "класс"]))
        self.assertTrue(is_unique_combination(self.entities_3, ["фамилия", "имя", "отчество", "школа"]))
        self.assertFalse(is_unique_combination(self.entities_3, ["фамилия", "имя", "класс"]))

    def test_minimal_cover_algorithm(self):
        self.assertEqual(len(minimal_cover_algorithm(self.entities_1, ["фамилия", "имя", "отчество", "класс"])), 2)
        self.assertEqual(len(minimal_cover_algorithm(self.entities_2, ["фамилия", "имя", "отчество", "класс", "школа"])), 1)
        self.assertEqual(len(minimal_cover_algorithm(self.entities_3, ["фамилия", "имя", "отчество", "класс", "школа"])), 2)

    def test_main(self):
        # Проверка только по количеству строк
        self.assertEqual(main(self.json_data_1).count('\n'), 2)
        self.assertEqual(main(self.json_data_2).count('\n'), 1)
        self.assertEqual(main(self.json_data_3).count('\n'), 2)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_input = sys.argv[1]
        result_csv = main(json_input)
        print(result_csv)
    else:
        unittest.main()