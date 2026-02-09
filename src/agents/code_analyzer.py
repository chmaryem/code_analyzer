

"""
Code Analyzer Agent - Agent d'analyse de code avec IA.
Analyse le code source avec Tree-sitter et utilise LangChain + Ollama pour l'analyse intelligente.
"""

import time
import os
import glob
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path

# Imports LangChain + Ollama (REQUIS)
from langchain_ollama import OllamaLLM
from langchain_classic.chains.llm import LLMChain
from langchain_classic.prompts import PromptTemplate


from ..models.analysis_result import (
     AnalysisResult,
    FunctionInfo,
    ClassInfo,
    CodeProblem,
    CodeSuggestion,
    ProblemSeverity,
    ProblemCategory
)
from ..utils.parser import CodeParser

class CodeAnalyzerAgent:
    """Agent principal pour l'analyse de code avec IA"""
    
    # Seuils de qualité du code
    MAX_FUNCTION_LENGTH = 50
    MAX_COMPLEXITY = 10
    MAX_NESTING_DEPTH = 4
    MAX_CLASS_METHODS = 15
    
    def __init__(self, ollama_model: str = "llama3.2", ollama_base_url: str = "http://localhost:11434"):
        """
        Initialise l'agent avec le parseur de code et le modèle Ollama
        
        Args:
            ollama_model: Nom du modèle Ollama à utiliser
            ollama_base_url: URL de base du serveur Ollama
        """
        self.parser = CodeParser()
        
        # Initialisation de Ollama via LangChain
        print(f"🤖 Initialisation de Ollama ({ollama_model})...")
        try:
            self.llm = OllamaLLM(
                model=ollama_model,
                base_url=ollama_base_url,
                temperature=0.3,  # Plus déterministe pour l'analyse de code
            )
            print(f"✅ Ollama initialisé avec succès")
        except Exception as e:
            print(f"⚠️  Avertissement: Ollama non disponible ({e})")
            print("   L'analyse continuera sans suggestions IA")
            self.llm = None
        
        # Prompt pour l'analyse de code
        self.analysis_prompt = PromptTemplate(
            input_variables=["code", "language", "metrics"],
            template="""Tu es un expert en analyse de code. Analyse le code suivant et fournis des recommandations.

Langage: {language}

Métriques détectées:
{metrics}

Code:
```{language}
{code}
```

Analyse ce code et fournis:
1. Les problèmes de qualité détectés (complexité, lisibilité, maintenabilité)
2. Les anti-patterns ou mauvaises pratiques
3. Les suggestions concrètes d'amélioration avec exemples de code
4. Un score de qualité global sur 10

Réponds de manière structurée et concise."""
        )
        
        if self.llm:
            self.analysis_chain = LLMChain(llm=self.llm, prompt=self.analysis_prompt)
        else:
            self.analysis_chain = None
    
    def analyze_file(self, file_path: str) -> AnalysisResult:
        """
        Analyse un fichier de code complet
        
        Args:
            file_path: Chemin du fichier à analyser
            
        Returns:
            Résultat complet de l'analyse
        """
        print(f"\n🔍 Analyse du fichier : {file_path}")
        
        # Parser le fichier
        ast_root = self.parser.parse_file(file_path)
        if not ast_root:
            return AnalysisResult(
                file_path=file_path,
                language="unknown",
                success=False,
                timestamp=datetime.now().isoformat()
            )
        
        # Lire le code source
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            print(f"❌ Erreur lors de la lecture du fichier : {e}")
            return AnalysisResult(
                file_path=file_path,
                language="unknown",
                success=False,
                timestamp=datetime.now().isoformat()
            )
        
        # Détecter le langage
        language = self.parser.detect_language(file_path) or "unknown"
        
        # Analyser le code
        return self._perform_analysis(file_path, code, language, ast_root)
    
    def analyze_files(self, file_paths: List[str]) -> List[AnalysisResult]:
        """
        Analyse plusieurs fichiers
        
        Args:
            file_paths: Liste des chemins de fichiers à analyser
            
        Returns:
            Liste des résultats d'analyse
        """
        print(f"\n📁 Analyse de {len(file_paths)} fichier(s)...")
        results = []
        
        for file_path in file_paths:
            result = self.analyze_file(file_path)
            results.append(result)
        
        return results
    
    def analyze_directory(self, directory_path: str, recursive: bool = True, 
                         extensions: Optional[List[str]] = None) -> List[AnalysisResult]:
        """
        Analyse tous les fichiers d'un répertoire
        
        Args:
            directory_path: Chemin du répertoire
            recursive: Si True, analyse récursivement les sous-dossiers
            extensions: Liste des extensions à analyser (ex: ['.py', '.js'])
                       Si None, utilise toutes les extensions supportées
            
        Returns:
            Liste des résultats d'analyse
        """
        if extensions is None:
            extensions = list(self.parser.LANGUAGE_MAP.keys())
        
        print(f"\n📂 Analyse du répertoire : {directory_path}")
        print(f"   Extensions recherchées : {', '.join(extensions)}")
        
        files_to_analyze = []
        
        if recursive:
            for ext in extensions:
                pattern = os.path.join(directory_path, '**', f'*{ext}')
                files_to_analyze.extend(glob.glob(pattern, recursive=True))
        else:
            for ext in extensions:
                pattern = os.path.join(directory_path, f'*{ext}')
                files_to_analyze.extend(glob.glob(pattern))
        
        print(f"   Fichiers trouvés : {len(files_to_analyze)}")
        
        return self.analyze_files(files_to_analyze)
    
    def analyze_code(self, code: str, language: str, filename: str = "<inline_code>") -> AnalysisResult:
        """
        Analyse une chaîne de code directement
        
        Args:
            code: Code source à analyser
            language: Langage de programmation
            filename: Nom du fichier (optionnel)
            
        Returns:
            Résultat complet de l'analyse
        """
        print(f"\n🔍 Analyse de code : {language}")
        
        # Parser le code
        ast_root = self.parser.parse_code(code, language)
        if not ast_root:
            return AnalysisResult(
                file_path=filename,
                language=language,
                success=False,
                timestamp=datetime.now().isoformat()
            )
        
        return self._perform_analysis(filename, code, language, ast_root)
    
    def _perform_analysis(self, file_path: str, code: str, language: str, ast_root: Any) -> AnalysisResult:
        """
        Effectue l'analyse complète du code
        """
        # Créer le résultat de base
        result = AnalysisResult(
            file_path=file_path,
            language=language,
            success=True,
            timestamp=datetime.now().isoformat()
        )
        
        # Extraire et analyser les fonctions
        functions_nodes = self.parser.extract_functions(ast_root)
        for func_node in functions_nodes:
            func_info = self._analyze_function(func_node, code, language)
            result.functions.append(func_info)
        
        # Extraire et analyser les classes
        classes_nodes = self.parser.extract_classes(ast_root)
        for class_node in classes_nodes:
            class_info = self._analyze_class(class_node, code, functions_nodes)
            result.classes.append(class_info)
        
        # Extraire les imports
        imports_nodes = self.parser.extract_imports(ast_root)
        for import_node in imports_nodes:
            import_text = self.parser.get_node_text(import_node, code)
            result.imports.append(import_text.strip())
        
        # Détecter le code dupliqué
        duplicates = self.parser.find_duplicate_code(ast_root, code)
        if duplicates:
            for dup in duplicates:
                problem = CodeProblem(
                    category=ProblemCategory.STYLE,
                    severity=ProblemSeverity.WARNING,
                    message=f"Code dupliqué détecté ({len(dup['text'])} caractères)",
                    location="Multiple emplacements",
                    details={"duplicate_count": len(duplicates)}
                )
                # Ajouter à la première fonction (simplifié)
                if result.functions:
                    result.functions[0].add_problem(problem)
        
        # Calculer les métriques globales
        result.calculate_metrics()
        
        # Analyse avec IA (si disponible)
        if self.analysis_chain and code.strip():
            self._ai_analysis(result, code, language)
        
        return result
    
    def _ai_analysis(self, result: AnalysisResult, code: str, language: str):
        """
        Effectue une analyse avec Ollama via LangChain
        """
        try:
            print("🤖 Analyse IA en cours...")
            
            # Préparer les métriques pour le prompt
            metrics_summary = f"""
- Fonctions: {result.metrics.total_functions}
- Classes: {result.metrics.total_classes}
- Problèmes détectés: {result.metrics.total_problems}
- Complexité moyenne: {sum(f.complexity for f in result.functions) / len(result.functions) if result.functions else 0:.1f}
"""
            
            # Limiter la taille du code pour l'analyse
            code_sample = code[:3000] + "\n..." if len(code) > 3000 else code
            
            # Exécuter l'analyse
            response = self.analysis_chain.invoke({
                "code": code_sample,
                "language": language,
                "metrics": metrics_summary
            })
            
            ai_text = response.get('text', '') if isinstance(response, dict) else str(response)
            
            # Créer un problème avec les suggestions IA
            if ai_text:
                ai_problem = CodeProblem(
                    category=ProblemCategory.STYLE,
                    severity=ProblemSeverity.INFO,
                    message="Analyse IA - Suggestions d'amélioration",
                    location="Global",
                    details={"ai_analysis": ai_text}
                )
                
                ai_suggestion = CodeSuggestion(
                    type="ai_improvement",
                    description="Améliorations suggérées par IA",
                    original_code=code_sample[:200] + "..." if len(code_sample) > 200 else code_sample,
                    suggested_code="Voir l'explication détaillée",
                    explanation=ai_text,
                    priority=2
                )
                ai_problem.add_suggestion(ai_suggestion)
                
                # Ajouter au premier élément disponible
                if result.functions:
                    result.functions[0].add_problem(ai_problem)
                elif result.classes:
                    result.classes[0].add_problem(ai_problem)
            
            print("✅ Analyse IA terminée")
            
        except Exception as e:
            print(f"⚠️  Erreur lors de l'analyse IA : {e}")
    
    def _analyze_function(self, func_node: Any, code: str, language: str) -> FunctionInfo:
        """
        Analyse une fonction individuelle
        """
        # Extraire les métriques de base
        func_text = self.parser.get_node_text(func_node, code)
        func_name = self._extract_function_name(func_node, code)
        line_count = self.parser.count_lines(func_node, code)
        complexity = self.parser.calculate_complexity(func_node)
        nesting_depth = self.parser.calculate_nesting_depth(func_node)
        
        start_line = func_node.start_point[0] + 1
        end_line = func_node.end_point[0] + 1
        
        # Créer l'objet FunctionInfo
        func_info = FunctionInfo(
            name=func_name,
            start_line=start_line,
            end_line=end_line,
            line_count=line_count,
            complexity=complexity,
            nesting_depth=nesting_depth
        )
        
        # Détecter les problèmes et générer des suggestions
        self._detect_function_problems(func_info, func_text, code, language)
        
        return func_info
    
    def _analyze_class(self, class_node: Any, code: str, functions_nodes: List[Any]) -> ClassInfo:
        """
        Analyse une classe individuelle
        """
        class_name = self._extract_class_name(class_node, code)
        start_line = class_node.start_point[0] + 1
        end_line = class_node.end_point[0] + 1
        
        # Compter les méthodes dans la classe
        method_count = 0
        for func_node in functions_nodes:
            if (class_node.start_byte <= func_node.start_byte and 
                func_node.end_byte <= class_node.end_byte):
                method_count += 1
        
        class_info = ClassInfo(
            name=class_name,
            start_line=start_line,
            end_line=end_line,
            method_count=method_count
        )
        
        self._detect_class_problems(class_info)
        
        return class_info
    
    def _detect_function_problems(self, func_info: FunctionInfo, func_text: str, 
                                  full_code: str, language: str):
        """Détecte les problèmes dans une fonction et génère des suggestions"""
        
        # 1. Fonction trop longue
        if func_info.line_count > self.MAX_FUNCTION_LENGTH:
            severity = ProblemSeverity.ERROR if func_info.line_count > 100 else ProblemSeverity.WARNING
            problem = CodeProblem(
                category=ProblemCategory.LENGTH,
                severity=severity,
                message=f"Fonction '{func_info.name}' trop longue ({func_info.line_count} lignes, max: {self.MAX_FUNCTION_LENGTH})",
                location=f"l. {func_info.start_line}-{func_info.end_line}",
                details={"current_length": func_info.line_count, "max_length": self.MAX_FUNCTION_LENGTH}
            )
            
            suggestion = CodeSuggestion(
                type="extract_function",
                description="Diviser en fonctions plus petites",
                original_code=func_text[:200] + "..." if len(func_text) > 200 else func_text,
                suggested_code=self._generate_refactor_suggestion(func_info.name, language),
                explanation=f"Les fonctions longues sont difficiles à comprendre. Divisez '{func_info.name}' en fonctions plus petites avec des responsabilités uniques.",
                priority=1
            )
            problem.add_suggestion(suggestion)
            func_info.add_problem(problem)
        
        # 2. Complexité élevée
        if func_info.complexity > self.MAX_COMPLEXITY:
            severity = ProblemSeverity.ERROR if func_info.complexity > 15 else ProblemSeverity.WARNING
            problem = CodeProblem(
                category=ProblemCategory.COMPLEXITY,
                severity=severity,
                message=f"Complexité élevée ({func_info.complexity}, max: {self.MAX_COMPLEXITY})",
                location=f"l. {func_info.start_line}-{func_info.end_line}",
                details={"current_complexity": func_info.complexity, "max_complexity": self.MAX_COMPLEXITY}
            )
            
            suggestion = CodeSuggestion(
                type="simplify_logic",
                description="Simplifier la logique conditionnelle",
                original_code=func_text[:200] + "..." if len(func_text) > 200 else func_text,
                suggested_code=self._generate_simplify_suggestion(func_info.name, language),
                explanation=f"Utilisez le pattern 'Early Return' ou extrayez les conditions complexes.",
                priority=1
            )
            problem.add_suggestion(suggestion)
            func_info.add_problem(problem)
        
        # 3. Imbrication profonde
        if func_info.nesting_depth > self.MAX_NESTING_DEPTH:
            severity = ProblemSeverity.ERROR if func_info.nesting_depth > 6 else ProblemSeverity.WARNING
            problem = CodeProblem(
                category=ProblemCategory.NESTING,
                severity=severity,
                message=f"Imbrication trop profonde ({func_info.nesting_depth} niveaux, max: {self.MAX_NESTING_DEPTH})",
                location=f"l. {func_info.start_line}-{func_info.end_line}",
                details={"current_depth": func_info.nesting_depth, "max_depth": self.MAX_NESTING_DEPTH}
            )
            
            suggestion = CodeSuggestion(
                type="reduce_nesting",
                description="Réduire l'imbrication",
                original_code=func_text[:200] + "..." if len(func_text) > 200 else func_text,
                suggested_code=self._generate_denest_suggestion(func_info.name, language),
                explanation="Utilisez 'continue', 'break', ou extrayez les blocs imbriqués.",
                priority=2
            )
            problem.add_suggestion(suggestion)
            func_info.add_problem(problem)
        
        # 4. Documentation manquante
        if not self._has_documentation(func_text):
            problem = CodeProblem(
                category=ProblemCategory.DOCUMENTATION,
                severity=ProblemSeverity.INFO,
                message=f"Fonction '{func_info.name}' sans documentation",
                location=f"l. {func_info.start_line}",
                details={}
            )
            
            suggestion = CodeSuggestion(
                type="add_documentation",
                description="Ajouter une documentation",
                original_code=f"def {func_info.name}():" if language == 'python' else f"function {func_info.name}()",
                suggested_code=self._generate_doc_suggestion(func_info.name, language),
                explanation="Ajoutez une documentation pour améliorer la maintenabilité.",
                priority=3
            )
            problem.add_suggestion(suggestion)
            func_info.add_problem(problem)
    
    def _detect_class_problems(self, class_info: ClassInfo):
        """Détecte les problèmes dans une classe"""
        if class_info.method_count > self.MAX_CLASS_METHODS:
            problem = CodeProblem(
                category=ProblemCategory.COMPLEXITY,
                severity=ProblemSeverity.WARNING,
                message=f"Classe '{class_info.name}' avec trop de méthodes ({class_info.method_count})",
                location=f"l. {class_info.start_line}-{class_info.end_line}",
                details={"current_methods": class_info.method_count, "max_methods": self.MAX_CLASS_METHODS}
            )
            
            suggestion = CodeSuggestion(
                type="split_class",
                description="Diviser la classe",
                original_code=f"class {class_info.name}:",
                suggested_code=f"# Diviser selon les responsabilités\nclass {class_info.name}Validator:\n    pass\n\nclass {class_info.name}Processor:\n    pass",
                explanation=f"Une classe avec {class_info.method_count} méthodes viole le principe de responsabilité unique.",
                priority=2
            )
            problem.add_suggestion(suggestion)
            class_info.add_problem(problem)
    
    def _generate_refactor_suggestion(self, func_name: str, language: str) -> str:
        """Génère une suggestion de refactorisation"""
        if language == 'python':
            return f"""def {func_name}_helper_1():
    # Logique extraite
    pass

def {func_name}_helper_2():
    # Autre logique extraite
    pass

def {func_name}():
    {func_name}_helper_1()
    {func_name}_helper_2()"""
        else:
            return f"""function {func_name}Helper1() {{
    // Logique extraite
}}

function {func_name}() {{
    {func_name}Helper1();
}}"""
    
    def _generate_simplify_suggestion(self, func_name: str, language: str) -> str:
        """Génère une suggestion de simplification"""
        if language == 'python':
            return f"""def {func_name}():
    if not condition_1:
        return
    if not condition_2:
        return
    # Logique principale"""
        else:
            return f"""function {func_name}() {{
    if (!condition1) return;
    if (!condition2) return;
    // Logique principale
}}"""
    
    def _generate_denest_suggestion(self, func_name: str, language: str) -> str:
        """Génère une suggestion de réduction d'imbrication"""
        if language == 'python':
            return f"""def {func_name}():
    for item in items:
        if not is_valid(item):
            continue
        process_item(item)"""
        else:
            return f"""function {func_name}() {{
    for (let item of items) {{
        if (!isValid(item)) continue;
        processItem(item);
    }}
}}"""
    
    def _generate_doc_suggestion(self, func_name: str, language: str) -> str:
        """Génère une suggestion de documentation"""
        if language == 'python':
            return f'''def {func_name}():
    """
    Description de la fonction.
    
    Args:
        arg1: Description
    
    Returns:
        Description du retour
    """'''
        else:
            return f'''/**
 * Description de la fonction
 * @param {{type}} param1 - Description
 * @returns {{type}} Description du retour
 */
function {func_name}() {{}}'''
    
    def _extract_function_name(self, func_node: Any, code: str) -> str:
        """Extrait le nom d'une fonction"""
        for child in func_node.children:
            if child.type == 'identifier' or child.type == 'property_identifier':
                return self.parser.get_node_text(child, code)
        return "<anonymous>"
    
    def _extract_class_name(self, class_node: Any, code: str) -> str:
        """Extrait le nom d'une classe"""
        for child in class_node.children:
            if child.type == 'identifier' or child.type == 'type_identifier':
                return self.parser.get_node_text(child, code)
        return "<anonymous>"
    
    def _has_documentation(self, func_text: str) -> bool:
        """Vérifie si une fonction a de la documentation"""
        lower_text = func_text.lower()
        return ('"""' in func_text or "'''" in func_text or 
                '/*' in func_text or '//' in func_text or
                func_text.strip().startswith('#'))
    
    def generate_summary_report(self, results: List[AnalysisResult]) -> str:
        """
        Génère un rapport résumé pour plusieurs analyses
        
        Args:
            results: Liste des résultats d'analyse
            
        Returns:
            Rapport formaté en texte
        """
        report = []
        report.append("=" * 80)
        report.append("📊 RAPPORT D'ANALYSE DE CODE")
        report.append("=" * 80)
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Fichiers analysés: {len(results)}")
        report.append("")
        
        total_problems = sum(r.metrics.total_problems for r in results)
        total_functions = sum(r.metrics.total_functions for r in results)
        total_classes = sum(r.metrics.total_classes for r in results)
        
        report.append("🔢 STATISTIQUES GLOBALES")
        report.append(f"  Fonctions: {total_functions}")
        report.append(f"  Classes: {total_classes}")
        report.append(f"  Problèmes: {total_problems}")
        report.append("")
        
        # Détails par fichier
        report.append("📁 DÉTAILS PAR FICHIER")
        for result in results:
            if result.success:
                status = "✅" if not result.has_problems() else "⚠️"
                report.append(f"{status} {result.file_path}")
                report.append(f"   Langage: {result.language}")
                report.append(f"   Problèmes: {result.metrics.total_problems}")
                if result.has_problems():
                    report.append(f"     - Critiques: {result.metrics.critical_problems}")
                    report.append(f"     - Erreurs: {result.metrics.error_problems}")
                    report.append(f"     - Avertissements: {result.metrics.warning_problems}")
            else:
                report.append(f"❌ {result.file_path} (échec)")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def print_analysis_summary(self, result: AnalysisResult):
        """Affiche un résumé détaillé de l'analyse"""
        print("\n" + "="*80)
        print(f"📊 RÉSUMÉ DE L'ANALYSE : {result.file_path}")
        print("="*80)
        print(f"Langage: {result.language}")
        print(f"Fonctions: {result.metrics.total_functions}")
        print(f"Classes: {result.metrics.total_classes}")
        print(f"\nProblèmes détectés: {result.metrics.total_problems}")
        print(f"  🔴 Critiques: {result.metrics.critical_problems}")
        print(f"  ❌ Erreurs: {result.metrics.error_problems}")
        print(f"  ⚠️  Avertissements: {result.metrics.warning_problems}")
        print(f"  ℹ️  Informations: {result.metrics.info_problems}")
        
        if result.has_problems():
            print("\n" + "="*80)
            print("📋 PROBLÈMES ET SUGGESTIONS")
            print("="*80)
            
            all_problems = result.get_all_problems()
            for idx, problem in enumerate(all_problems, 1):
                print(f"\n{idx}. {problem}")
                if problem.details:
                    print(f"   Détails: {problem.details}")
                
                if problem.has_suggestions():
                    print(f"\n   💡 Suggestions:")
                    for i, suggestion in enumerate(problem.suggestions, 1):
                        print(f"\n   {i}. [{suggestion.type.upper()}] {suggestion.description}")
                        print(f"      Priorité: {'Haute' if suggestion.priority == 1 else 'Moyenne' if suggestion.priority == 2 else 'Basse'}")
                        print(f"\n      Explication:")
                        print(f"      {suggestion.explanation}")
        else:
            print("\n🎉 Aucun problème détecté !")
        
        print("="*80)