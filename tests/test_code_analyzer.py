"""
Tests pour le Code Analyzer Agent.
"""

import sys
import os

# Ajouter le répertoire src au path
# Ajouter le répertoire racine du projet au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.code_analyzer import CodeAnalyzerAgent
from src.models.analysis_result import AnalysisResult

def print_separator(char="=", length=80):
    """Affiche un séparateur"""
    print("\n" + char * length)


def test_python_code():
    """Test 1: Analyse de code Python"""
    print_separator()
    print("TEST 1: Analyse de code Python")
    print_separator()
    
    try:
        agent = CodeAnalyzerAgent()
        
        # Code Python avec plusieurs problèmes
        python_code = '''
def process_data(data, config, options, flags, settings, params):
    """Function with too many parameters and complexity"""
    if data is None:
        return None
    if len(data) == 0:
        return None
    if type(data) != list:
        return None
    if data[0] is None:
        return None
    
    result = []
    for item in data:
        if item is not None:
            if item.get('value') is not None:
                if item.get('value') > 0:
                    if item.get('value') < 100:
                        if item.get('name') is not None:
                            if item.get('name') != '':
                                if config.get('enabled'):
                                    result.append(item)
    return result

class HugeClass:
    def method1(self): pass
    def method2(self): pass
    def method3(self): pass
    def method4(self): pass
    def method5(self): pass
    def method6(self): pass
    def method7(self): pass
    def method8(self): pass
    def method9(self): pass
    def method10(self): pass
    def method11(self): pass
    def method12(self): pass
    def method13(self): pass
    def method14(self): pass
    def method15(self): pass
    def method16(self): pass
    def method17(self): pass
'''
        
        result = agent.analyze_code(python_code, 'python', 'test_python.py')
        
        if result.success:
            print(f"✅ Analyse réussie")
            agent.print_analysis_summary(result)
            
            suggestions = result.get_all_suggestions()
            print(f"\n💡 {len(suggestions)} suggestions générées")
            
            return True
        else:
            print(f"❌ Analyse échouée")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_javascript_code():
    """Test 2: Analyse de code JavaScript"""
    print_separator()
    print("TEST 2: Analyse de code JavaScript")
    print_separator()
    
    try:
        agent = CodeAnalyzerAgent()
        
        js_code = '''
function complexFunction(a, b, c, d, e, f) {
    if (a === null) {
        return null;
    }
    
    if (b === undefined) {
        return null;
    }
    
    for (let i = 0; i < 100; i++) {
        for (let j = 0; j < 100; j++) {
            for (let k = 0; k < 100; k++) {
                if (i > 0) {
                    if (j > 0) {
                        if (k > 0) {
                            console.log(i, j, k);
                        }
                    }
                }
            }
        }
    }
    
    return true;
}

class BigClass {
    method1() {}
    method2() {}
    method3() {}
    method4() {}
    method5() {}
    method6() {}
    method7() {}
    method8() {}
    method9() {}
    method10() {}
    method11() {}
    method12() {}
    method13() {}
    method14() {}
    method15() {}
    method16() {}
}
'''
        
        result = agent.analyze_code(js_code, 'javascript', 'test.js')
        
        if result.success:
            print(f"✅ Analyse JavaScript réussie")
            agent.print_analysis_summary(result)
            return True
        else:
            print(f"❌ Analyse échouée")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_java_code():
    """Test 3: Analyse de code Java"""
    print_separator()
    print("TEST 3: Analyse de code Java")
    print_separator()
    
    try:
        agent = CodeAnalyzerAgent()
        
        java_code = '''
public class Calculator {
    public int complexCalculation(int a, int b, int c) {
        int result = 0;
        
        if (a > 0) {
            if (b > 0) {
                if (c > 0) {
                    for (int i = 0; i < a; i++) {
                        for (int j = 0; j < b; j++) {
                            for (int k = 0; k < c; k++) {
                                result += i + j + k;
                            }
                        }
                    }
                }
            }
        }
        
        return result;
    }
}
'''
        
        result = agent.analyze_code(java_code, 'java', 'Calculator.java')
        
        if result.success:
            print(f"✅ Analyse Java réussie")
            agent.print_analysis_summary(result)
            return True
        else:
            print(f"❌ Analyse échouée")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_files():
    """Test 4: Analyse de plusieurs fichiers"""
    print_separator()
    print("TEST 4: Analyse de plusieurs fichiers")
    print_separator()
    
    try:
        agent = CodeAnalyzerAgent()
        
        # Créer des fichiers temporaires
        test_files = []
        
        # Fichier Python
        py_file = '/tmp/test_python.py'
        with open(py_file, 'w') as f:
            f.write('''
def simple_function():
    """A simple function"""
    return True

def complex_function():
    for i in range(100):
        for j in range(100):
            if i > 0:
                if j > 0:
                    print(i, j)
''')
        test_files.append(py_file)
        
        # Fichier JavaScript
        js_file = '/tmp/test_script.js'
        with open(js_file, 'w') as f:
            f.write('''
function simpleFunc() {
    return true;
}

function nestedFunc() {
    for (let i = 0; i < 10; i++) {
        for (let j = 0; j < 10; j++) {
            for (let k = 0; k < 10; k++) {
                console.log(i, j, k);
            }
        }
    }
}
''')
        test_files.append(js_file)
        
        # Analyser tous les fichiers
        results = agent.analyze_files(test_files)
        
        print(f"✅ {len(results)} fichiers analysés")
        
        # Générer un rapport
        report = agent.generate_summary_report(results)
        print(report)
        
        # Nettoyer
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_directory_analysis():
    """Test 5: Analyse d'un répertoire"""
    print_separator()
    print("TEST 5: Analyse d'un répertoire")
    print_separator()
    
    try:
        agent = CodeAnalyzerAgent()
        
        # Créer un répertoire de test
        test_dir = '/tmp/test_code_analysis'
        os.makedirs(test_dir, exist_ok=True)
        
        # Créer des sous-dossiers
        os.makedirs(os.path.join(test_dir, 'subdir'), exist_ok=True)
        
        # Créer des fichiers
        files = {
            'script1.py': 'def func1():\n    pass\n',
            'script2.py': 'def func2():\n    for i in range(100):\n        for j in range(100):\n            pass\n',
            'app.js': 'function test() { return true; }\n',
            'subdir/utils.py': 'def helper():\n    return None\n',
        }
        
        for filename, content in files.items():
            filepath = os.path.join(test_dir, filename)
            with open(filepath, 'w') as f:
                f.write(content)
        
        # Analyser le répertoire
        results = agent.analyze_directory(test_dir, recursive=True)
        
        print(f"✅ Répertoire analysé: {len(results)} fichiers trouvés")
        
        # Afficher le rapport
        report = agent.generate_summary_report(results)
        print(report)
        
        # Nettoyer
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ai_analysis():
    """Test 6: Test de l'analyse IA (si Ollama est disponible)"""
    print_separator()
    print("TEST 6: Test de l'analyse IA avec Ollama")
    print_separator()
    
    try:
        agent = CodeAnalyzerAgent()
        
        if agent.llm is None:
            print("⚠️  Ollama non disponible - Test ignoré")
            return True
        
        code = '''
def poorly_written_function(x, y, z):
    result = 0
    for i in range(100):
        for j in range(100):
            for k in range(100):
                if i > 0:
                    if j > 0:
                        if k > 0:
                            result += x + y + z
    return result
'''
        
        result = agent.analyze_code(code, 'python')
        
        if result.success:
            print(f"✅ Analyse IA effectuée")
            
            # Rechercher les suggestions IA
            ai_suggestions = [s for s in result.get_all_suggestions() if s.type == 'ai_improvement']
            
            if ai_suggestions:
                print(f"🤖 {len(ai_suggestions)} suggestion(s) IA générée(s)")
                for suggestion in ai_suggestions:
                    print(f"\nSuggestion IA:")
                    print(suggestion.explanation[:500] + "..." if len(suggestion.explanation) > 500 else suggestion.explanation)
            
            return True
        else:
            print(f"❌ Analyse échouée")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Fonction principale de test"""
    print("\n" + "="*80)
    print("🧪 TESTS DU CODE ANALYZER AGENT - VERSION COMPLÈTE")
    print("="*80)
    
    tests = [
        ("Analyse de code Python", test_python_code),
        ("Analyse de code JavaScript", test_javascript_code),
        ("Analyse de code Java", test_java_code),
        ("Analyse de plusieurs fichiers", test_multiple_files),
        ("Analyse d'un répertoire", test_directory_analysis),
        ("Test de l'analyse IA", test_ai_analysis),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erreur lors du test '{test_name}': {e}")
            results.append((test_name, False))
    
    # Résumé
    print_separator()
    print("📊 RÉSUMÉ DES TESTS")
    print_separator()
    
    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed
    
    for test_name, result in results:
        status = "✅" if result else "❌"
        print(f"{status} {test_name}")
    
    print(f"\n✅ Tests réussis: {passed}/{len(results)}")
    print(f"❌ Tests échoués: {failed}/{len(results)}")
    print_separator()
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)