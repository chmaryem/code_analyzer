"""
Exemple de code Python pour tester l'agent d'analyse.
Ce code contient intentionnellement quelques problèmes.
"""

import os
import sys


def calculate_average(numbers):
    """Calcule la moyenne d'une liste de nombres."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)


def process_data(data):
    """
    Fonction trop longue avec complexité élevée.
    Cette fonction démontre plusieurs problèmes potentiels.
    """
    results = []
    
    for item in data:
        if item['status'] == 'active':
            for key, value in item.items():
                if isinstance(value, str):
                    for word in value.split():
                        if word.startswith('test'):
                            if len(word) > 5:
                                if 'error' in word:
                                    results.append({
                                        'item': item,
                                        'word': word,
                                        'processed': True
                                    })
                elif isinstance(value, int):
                    if value > 0:
                        for i in range(value):
                            if i % 2 == 0:
                                results.append({
                                    'item': item,
                                    'value': i,
                                    'processed': False
                                })
        elif item['status'] == 'inactive':
            # Traitement des items inactifs
            if 'archive' in item:
                if item['archive'] == True:
                    for archive_item in item['archive']:
                        results.append({
                            'item': item,
                            'archive_item': archive_item,
                            'status': 'archived'
                        })
    
    return results


class DataProcessor:
    """Classe pour traiter des données."""
    
    def __init__(self, config):
        self.config = config
        self.cache = {}
    
    def process(self, data):
        """Traite les données."""
        return process_data(data)
