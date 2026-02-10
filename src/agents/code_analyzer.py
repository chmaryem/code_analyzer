"""
Code Analyzer Agent - version améliorée:
- Entrées: fichier / dossier / code inline
- Détection erreurs: Tree-sitter ERROR nodes + SyntaxError python
- IA: réponse JSON + patchs + code corrigé
- LangChain: prompt | llm 
"""

import os
import glob
import json
from typing import Optional, List, Any, Dict
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

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


#Le LLM doit retourner un JSON structuré, pas du texte libre
class AIFixPatch(BaseModel):
    title: str = Field(...)
    unified_diff: str = Field(..., description="Patch format unified diff")
    rationale: str = Field(default="")

class AIIssue(BaseModel):
    """Un problème détecté par l'IA"""
    category: str = Field(...)
    severity: str = Field(...)
    message: str = Field(...)
    location: str = Field(default="Global")

#Utilité : Valider automatiquement la réponse JSON du LLM.
class AIResponse(BaseModel):
    """Réponse complète de l'IA (structure attendue)"""
    issues: List[AIIssue] = Field(default_factory=list)
    patches: List[AIFixPatch] = Field(default_factory=list)
    fixed_code: Optional[str] = None
    score: Optional[float] = None


class CodeAnalyzerAgent:
    MAX_FUNCTION_LENGTH = 50
    MAX_COMPLEXITY = 10
    MAX_NESTING_DEPTH = 4
    MAX_CLASS_METHODS = 15

    def __init__(self, ollama_model: str = "llama3.2", ollama_base_url: str = "http://localhost:11434"):
        self.parser = CodeParser()

        print(f" Initialisation de Ollama ({ollama_model})...")
        try:
            self.llm = OllamaLLM(
                model=ollama_model,
                base_url=ollama_base_url,
                temperature=0.2,
            )
            print("Ollama initialisé avec succès")
        except Exception as e:
            print(f" Avertissement: Ollama non disponible ({e})")
            print("  L'analyse continuera sans suggestions IA")
            self.llm = None

        # Prompt IA -> JSON strict
        self.analysis_prompt = PromptTemplate.from_template(
            """Tu es un expert senior en revue de code et refactoring.

But: détecter bugs/erreurs, anti-patterns, risques sécurité/perf, et proposer des corrections.

Retourne UNIQUEMENT du JSON valide (pas de markdown, pas de texte autour),
avec ce schéma:
{{
  "issues": [
    {{"category": "security|bug|style|performance|maintainability|syntax", "severity": "info|warning|error|critical", "message": "...", "location": "l.X:Y - l.A:B"}}
  ],
  "patches": [
    {{"title": "...", "unified_diff": "...", "rationale": "..."}}
  ],
  "fixed_code": "OPTIONNEL: code complet corrigé si court",
  "score": 0-10
}}

Langage: {language}

Métriques:
{metrics}

Code:
{code}
"""
        )

        self.analysis_chain: Optional[RunnableSequence] = None
        if self.llm:
            # prompt -> llm
            self.analysis_chain = self.analysis_prompt | self.llm

    
    #  Entrées utilisateur
    
    def analyze_path(self, path: str, recursive: bool = True, extensions: Optional[List[str]] = None) -> List[AnalysisResult]:
        """
        Analyse un fichier ou un dossier (auto-détection).
        """
        if os.path.isdir(path):
            # C'est un dossier → Analyser tous les fichiers
            return self.analyze_directory(path, recursive=recursive, extensions=extensions)
         # C'est un fichier → Analyser ce fichier seul
        return [self.analyze_file(path)]
    



    def analyze_file(self, file_path: str) -> AnalysisResult:
        print(f"\n Analyse du fichier : {file_path}")

 # 1. PARSER LE FICHIER
        ast_root = self.parser.parse_file(file_path)
        if not ast_root:
            return AnalysisResult(file_path=file_path, language="unknown", success=False, timestamp=datetime.now().isoformat())

 # 2. LIRE LE CONTENU
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
        except Exception as e:
            print(f" Erreur lecture fichier : {e}")
            return AnalysisResult(file_path=file_path, language="unknown", success=False, timestamp=datetime.now().isoformat())
#3. DÉTECTER LE LANGAGE
        language = self.parser.detect_language(file_path) or "unknown"
#4. ANALYSER        
        return self._perform_analysis(file_path, code, language, ast_root)
    
    


    def analyze_directory(self, directory_path: str, recursive: bool = True, extensions: Optional[List[str]] = None) -> List[AnalysisResult]:
        if extensions is None:
            extensions = list(self.parser.LANGUAGE_MAP.keys())

        print(f"\n Analyse du répertoire : {directory_path}")
        print(f" Extensions recherchées : {', '.join(extensions)}")
 # 2. RECHERCHE DE FICHIERS
        files_to_analyze: List[str] = []
        if recursive:
            # Récursif : chercher dans sous-dossiers
            for ext in extensions:
                pattern = os.path.join(directory_path, '**', f'*{ext}')
                files_to_analyze.extend(glob.glob(pattern, recursive=True))
        else:
            # Non récursif : seulement le dossier courant
            for ext in extensions:
                pattern = os.path.join(directory_path, f'*{ext}')
                files_to_analyze.extend(glob.glob(pattern))

        print(f"   Fichiers trouvés : {len(files_to_analyze)}")
        return self.analyze_files(files_to_analyze)

    def analyze_files(self, file_paths: List[str]) -> List[AnalysisResult]:
        print(f"\n Analyse de {len(file_paths)} fichier(s)...")
           # 3. ANALYSER TOUS LES FICHIERS
        return [self.analyze_file(p) for p in file_paths]


    def analyze_code(self, code: str, language: str, filename: str = "<inline_code>") -> AnalysisResult:
        print(f"\n Analyse de code : {language}")
 #1. PARSER LE CODE 
        ast_root = self.parser.parse_code(code, language)
        if not ast_root:
            return AnalysisResult(file_path=filename,
                                  language=language,
                                  success=False,
                                  timestamp=datetime.now().isoformat())
 #2. ANALYSER
        return self._perform_analysis(filename, code, language, ast_root)

   
    #  Analyse complète
    # C'est le cœur de l'agent. Orchestration complète.
    def _perform_analysis(self, file_path: str, code: str, language: str, ast_root: Any) -> AnalysisResult:
         # 1. CRÉER LE RÉSULTAT
        result = AnalysisResult(
            file_path=file_path,
            language=language,
            success=True,
            timestamp=datetime.now().isoformat()
        )

        #  1) Erreurs syntaxe Tree-sitter
        if self.parser.has_parse_error(ast_root):
            for e in self.parser.get_parse_errors(ast_root, code):
                result.functions  # touch to keep structure
                problem = CodeProblem(
                    category=ProblemCategory.STYLE,
                    severity=ProblemSeverity.ERROR,
                    message=e["message"],
                    location=e["location"],
                    details={"snippet": e["snippet"]}
                )
                # attacher au resultat
                if result.functions:
                    result.functions[0].add_problem(problem)
                else:
                    # si aucune fonction, créer une pseudo-fonction globale
                    global_fn = FunctionInfo(
                        name="<global>",
                        start_line=1,
                        end_line=max(1, code.count("\n") + 1),
                        line_count=max(1, len(code.splitlines())),
                        complexity=1,
                        nesting_depth=1,
                        problems=[problem]
                    )
                    result.functions.append(global_fn)

        #  2) Erreurs Python réelles
        if language == "python":
            py_err = self._python_syntax_check(code, file_path)
            if py_err:
                problem = CodeProblem(
                    category=ProblemCategory.STYLE,
                    severity=ProblemSeverity.ERROR,
                    message=f"SyntaxError: {py_err['message']}",
                    location=py_err["location"],
                    details={"line": py_err.get("line", ""), "offset": py_err.get("offset", None)}
                )
                if result.functions:
                    result.functions[0].add_problem(problem)
                else:
                    global_fn = FunctionInfo(
                        name="<global>",
                        start_line=1,
                        end_line=max(1, code.count("\n") + 1),
                        line_count=max(1, len(code.splitlines())),
                        complexity=1,
                        nesting_depth=1,
                        problems=[problem]
                    )
                    result.functions.append(global_fn)

        # Fonctions/classes/imports + métriques (comme avant)
        functions_nodes = self.parser.extract_functions(ast_root)
        for func_node in functions_nodes:
            func_info = self._analyze_function(func_node, code, language)
            result.functions.append(func_info)

        classes_nodes = self.parser.extract_classes(ast_root)
        for class_node in classes_nodes:
            class_info = self._analyze_class(class_node, code, functions_nodes)
            result.classes.append(class_info)

        imports_nodes = self.parser.extract_imports(ast_root)
        for import_node in imports_nodes:
            result.imports.append(self.parser.get_node_text(import_node, code).strip())

        duplicates = self.parser.find_duplicate_code(ast_root, code)
        if duplicates and result.functions:
            problem = CodeProblem(
                category=ProblemCategory.STYLE,
                severity=ProblemSeverity.WARNING,
                message=f"Code dupliqué détecté ({len(duplicates)} duplications)",
                location="Multiple emplacements",
                details={"duplicate_count": len(duplicates)}
            )
            result.functions[0].add_problem(problem)

        result.calculate_metrics()

        #  IA : issues + patchs + code corrigé
        if self.analysis_chain and code.strip():
            self._ai_analysis_structured(result, code, language)

        return result

    def _python_syntax_check(self, code: str, filename: str) -> Optional[Dict[str, Any]]:
        try:
            compile(code, filename, "exec")
            return None
        except SyntaxError as e:
            line = (e.text or "").rstrip("\n")
            location = f"l.{e.lineno}:{e.offset}" if e.lineno and e.offset else "inconnu"
            return {"message": str(e.msg), "location": location, "line": line, "offset": e.offset}
        except Exception:
            return None

    def _ai_analysis_structured(self, result: AnalysisResult, code: str, language: str):
        try:
            print(" Analyse IA (JSON + patchs) en cours...")

            metrics_summary = f"""
- Fonctions: {result.metrics.total_functions}
- Classes: {result.metrics.total_classes}
- Problèmes détectés (heuristiques): {result.metrics.total_problems}
- Complexité moyenne: {sum(f.complexity for f in result.functions) / len(result.functions) if result.functions else 0:.1f}
"""

            code_sample = code[:6000] + "\n..." if len(code) > 6000 else code

            raw = self.analysis_chain.invoke({
                "code": code_sample,
                "language": language,
                "metrics": metrics_summary
            })

            text = raw if isinstance(raw, str) else str(raw)

            # Extra: nettoyer si jamais le modèle renvoie du texte autour
            text_stripped = text.strip()
            # essayer de trouver le premier '{' et dernier '}'
            if "{" in text_stripped and "}" in text_stripped:
                text_stripped = text_stripped[text_stripped.find("{"): text_stripped.rfind("}") + 1]

            ai = AIResponse.model_validate_json(text_stripped)

            # convertir issues -> CodeProblem
            for issue in ai.issues:
                sev = issue.severity.lower()
                cat = issue.category.lower()

                severity = {
                    "info": ProblemSeverity.INFO,
                    "warning": ProblemSeverity.WARNING,
                    "error": ProblemSeverity.ERROR,
                    "critical": ProblemSeverity.CRITICAL
                }.get(sev, ProblemSeverity.INFO)

                category = ProblemCategory.STYLE
                if "security" in cat:
                    category = ProblemCategory.SECURITY
                elif "performance" in cat:
                    category = ProblemCategory.PERFORMANCE
                elif "maintain" in cat:
                    category = ProblemCategory.COMPLEXITY
                elif "bug" in cat or "syntax" in cat:
                    category = ProblemCategory.STYLE

                problem = CodeProblem(
                    category=category,
                    severity=severity,
                    message=f"IA: {issue.message}",
                    location=issue.location,
                    details={}
                )
                if result.functions:
                    result.functions[0].add_problem(problem)

            # patchs -> suggestions
            if ai.patches:
                problem = CodeProblem(
                    category=ProblemCategory.STYLE,
                    severity=ProblemSeverity.INFO,
                    message="IA: Correctifs proposés (patchs)",
                    location="Global",
                    details={"patch_count": len(ai.patches)}
                )
                for p in ai.patches:
                    suggestion = CodeSuggestion(
                        type="ai_patch",
                        description=p.title,
                        original_code=code_sample[:200] + "..." if len(code_sample) > 200 else code_sample,
                        suggested_code=p.unified_diff,
                        explanation=p.rationale,
                        priority=1
                    )
                    problem.add_suggestion(suggestion)

                if result.functions:
                    result.functions[0].add_problem(problem)

            # fixed_code
            if ai.fixed_code:
                problem = CodeProblem(
                    category=ProblemCategory.STYLE,
                    severity=ProblemSeverity.INFO,
                    message="IA: Version corrigée proposée",
                    location="Global",
                    details={}
                )
                suggestion = CodeSuggestion(
                    type="ai_fixed_code",
                    description="Code complet corrigé (si applicable)",
                    original_code=code_sample[:400] + "..." if len(code_sample) > 400 else code_sample,
                    suggested_code=ai.fixed_code,
                    explanation="Le modèle propose une version corrigée complète.",
                    priority=1
                )
                problem.add_suggestion(suggestion)
                if result.functions:
                    result.functions[0].add_problem(problem)

            print(" Analyse IA terminée")

        except ValidationError as e:
            print(" IA a renvoyé un JSON invalide, fallback texte.")
            self._ai_analysis_fallback_text(result, code, language, extra_error=str(e))
        except Exception as e:
            print(f"  Erreur analyse IA : {e}")

    def _ai_analysis_fallback_text(self, result: AnalysisResult, code: str, language: str, extra_error: str = ""):
        # fallback minimal, pour ne pas casser
        if not self.analysis_chain:
            return
        try:
            metrics_summary = f"- Fonctions: {result.metrics.total_functions}\n- Classes: {result.metrics.total_classes}\n"
            code_sample = code[:3000] + "\n..." if len(code) > 3000 else code
            raw = self.analysis_chain.invoke({"code": code_sample, "language": language, "metrics": metrics_summary})
            text = raw if isinstance(raw, str) else str(raw)
            problem = CodeProblem(
                category=ProblemCategory.STYLE,
                severity=ProblemSeverity.INFO,
                message="Analyse IA (fallback texte)",
                location="Global",
                details={"ai_text": text[:2000], "json_error": extra_error[:500]}
            )
            suggestion = CodeSuggestion(
                type="ai_improvement",
                description="Suggestions IA (texte)",
                original_code=code_sample[:200] + "..." if len(code_sample) > 200 else code_sample,
                suggested_code="Voir details.ai_text",
                explanation=text,
                priority=2
            )
            problem.add_suggestion(suggestion)
            if result.functions:
                result.functions[0].add_problem(problem)
        except Exception:
            pass

    # -------------------------
    # Analyse métriques (comme ton code)
    # -------------------------
    def _analyze_function(self, func_node: Any, code: str, language: str) -> FunctionInfo:
        func_text = self.parser.get_node_text(func_node, code)
        func_name = self._extract_function_name(func_node, code)
        line_count = self.parser.count_lines(func_node, code)
        complexity = self.parser.calculate_complexity(func_node)
        nesting_depth = self.parser.calculate_nesting_depth(func_node)

        start_line = func_node.start_point[0] + 1
        end_line = func_node.end_point[0] + 1

        func_info = FunctionInfo(
            name=func_name,
            start_line=start_line,
            end_line=end_line,
            line_count=line_count,
            complexity=complexity,
            nesting_depth=nesting_depth
        )

        self._detect_function_problems(func_info, func_text, language)
        return func_info

    def _analyze_class(self, class_node: Any, code: str, functions_nodes: List[Any]) -> ClassInfo:
        class_name = self._extract_class_name(class_node, code)
        start_line = class_node.start_point[0] + 1
        end_line = class_node.end_point[0] + 1

        method_count = 0
        for func_node in functions_nodes:
            if class_node.start_byte <= func_node.start_byte and func_node.end_byte <= class_node.end_byte:
                method_count += 1

        class_info = ClassInfo(
            name=class_name,
            start_line=start_line,
            end_line=end_line,
            method_count=method_count
        )
        self._detect_class_problems(class_info)
        return class_info

    def _detect_function_problems(self, func_info: FunctionInfo, func_text: str, language: str):
        if func_info.line_count > self.MAX_FUNCTION_LENGTH:
            severity = ProblemSeverity.ERROR if func_info.line_count > 100 else ProblemSeverity.WARNING
            problem = CodeProblem(
                category=ProblemCategory.LENGTH,
                severity=severity,
                message=f"Fonction '{func_info.name}' trop longue ({func_info.line_count} lignes, max: {self.MAX_FUNCTION_LENGTH})",
                location=f"l. {func_info.start_line}-{func_info.end_line}",
                details={"current_length": func_info.line_count, "max_length": self.MAX_FUNCTION_LENGTH}
            )
            problem.add_suggestion(CodeSuggestion(
                type="extract_function",
                description="Diviser en fonctions plus petites",
                original_code=func_text[:250] + "..." if len(func_text) > 250 else func_text,
                suggested_code="Découper en helpers (responsabilités uniques).",
                explanation="Réduire longueur => lisibilité + testabilité.",
                priority=1
            ))
            func_info.add_problem(problem)

        if func_info.complexity > self.MAX_COMPLEXITY:
            severity = ProblemSeverity.ERROR if func_info.complexity > 15 else ProblemSeverity.WARNING
            problem = CodeProblem(
                category=ProblemCategory.COMPLEXITY,
                severity=severity,
                message=f"Complexité élevée ({func_info.complexity}, max: {self.MAX_COMPLEXITY})",
                location=f"l. {func_info.start_line}-{func_info.end_line}",
                details={"current_complexity": func_info.complexity, "max_complexity": self.MAX_COMPLEXITY}
            )
            problem.add_suggestion(CodeSuggestion(
                type="simplify_logic",
                description="Simplifier conditions (early returns / extraction)",
                original_code=func_text[:250] + "..." if len(func_text) > 250 else func_text,
                suggested_code="Appliquer early-return + extraire des prédicats.",
                explanation="Moins de branches => moins de bugs.",
                priority=1
            ))
            func_info.add_problem(problem)

        if func_info.nesting_depth > self.MAX_NESTING_DEPTH:
            severity = ProblemSeverity.ERROR if func_info.nesting_depth > 6 else ProblemSeverity.WARNING
            problem = CodeProblem(
                category=ProblemCategory.NESTING,
                severity=severity,
                message=f"Imbrication trop profonde ({func_info.nesting_depth} niveaux, max: {self.MAX_NESTING_DEPTH})",
                location=f"l. {func_info.start_line}-{func_info.end_line}",
                details={"current_depth": func_info.nesting_depth, "max_depth": self.MAX_NESTING_DEPTH}
            )
            problem.add_suggestion(CodeSuggestion(
                type="reduce_nesting",
                description="Réduire l'imbrication",
                original_code=func_text[:250] + "..." if len(func_text) > 250 else func_text,
                suggested_code="Utiliser continue/break + extraire blocs.",
                explanation="Moins d'imbrication => plus lisible.",
                priority=2
            ))
            func_info.add_problem(problem)

        if not self._has_documentation(func_text):
            problem = CodeProblem(
                category=ProblemCategory.DOCUMENTATION,
                severity=ProblemSeverity.INFO,
                message=f"Fonction '{func_info.name}' sans documentation",
                location=f"l. {func_info.start_line}",
                details={}
            )
            problem.add_suggestion(CodeSuggestion(
                type="add_documentation",
                description="Ajouter docstring / JSDoc",
                original_code=func_text.splitlines()[0] if func_text.splitlines() else "",
                suggested_code="Ajouter doc + exemples d'usage.",
                explanation="La doc réduit les erreurs d'usage.",
                priority=3
            ))
            func_info.add_problem(problem)

    def _detect_class_problems(self, class_info: ClassInfo):
        if class_info.method_count > self.MAX_CLASS_METHODS:
            problem = CodeProblem(
                category=ProblemCategory.COMPLEXITY,
                severity=ProblemSeverity.WARNING,
                message=f"Classe '{class_info.name}' avec trop de méthodes ({class_info.method_count})",
                location=f"l. {class_info.start_line}-{class_info.end_line}",
                details={"current_methods": class_info.method_count, "max_methods": self.MAX_CLASS_METHODS}
            )
            problem.add_suggestion(CodeSuggestion(
                type="split_class",
                description="Diviser la classe selon responsabilités",
                original_code=f"class {class_info.name}",
                suggested_code="Créer plusieurs classes/services (SRP).",
                explanation="SRP => code maintenable.",
                priority=2
            ))
            class_info.add_problem(problem)

    def _extract_function_name(self, func_node: Any, code: str) -> str:
        for child in func_node.children:
            if child.type in ('identifier', 'property_identifier'):
                return self.parser.get_node_text(child, code)
        return "<anonymous>"

    def _extract_class_name(self, class_node: Any, code: str) -> str:
        for child in class_node.children:
            if child.type in ('identifier', 'type_identifier'):
                return self.parser.get_node_text(child, code)
        return "<anonymous>"

    def _has_documentation(self, func_text: str) -> bool:
        return ('"""' in func_text or "'''" in func_text or '/*' in func_text or '//' in func_text or func_text.strip().startswith('#'))

    # tes méthodes report/print_summary peuvent rester identiques si tu veux
        # -------------------------
    # ✅ AFFICHAGE / RAPPORTS (pour CLI)
    # -------------------------
    def print_analysis_summary(self, result: AnalysisResult) -> None:
        """Affiche un résumé détaillé de l'analyse (compatible avec ton CLI)."""

        print("\n" + "=" * 80)
        print(f" RÉSUMÉ DE L'ANALYSE : {Path(result.file_path).name}")
        print("=" * 80)

        print(f"Langage: {result.language}")
        print(f"Fonctions: {result.metrics.total_functions}")
        print(f"Classes: {result.metrics.total_classes}")

        print(f"\nProblèmes détectés: {result.metrics.total_problems}")
        print(f"  Critiques: {result.metrics.critical_problems}")
        print(f"   Erreurs: {result.metrics.error_problems}")
        print(f"    Avertissements: {result.metrics.warning_problems}")
        print(f"   Informations: {result.metrics.info_problems}")

        if not result.has_problems():
            print("\n Aucun problème détecté !")
            print("=" * 80)
            return

        print("\n" + "=" * 80)
        print(" PROBLÈMES ET SUGGESTIONS")
        print("=" * 80)

        all_problems = result.get_all_problems()
        for idx, problem in enumerate(all_problems, 1):
            # Exemple d'affichage propre
            sev_icon = {
                ProblemSeverity.CRITICAL: "",
                ProblemSeverity.ERROR: "",
                ProblemSeverity.WARNING: "",
                ProblemSeverity.INFO: "ℹ",
            }.get(problem.severity, "ℹ")

            print(f"\n{idx}. {sev_icon} {problem.message} ({problem.category.value})")
            if problem.location:
                print(f"    {problem.location}")

            if problem.details:
                print(f"   Détails: {problem.details}")

            if problem.has_suggestions():
                print("\n   Suggestions:")
                for i, s in enumerate(problem.suggestions, 1):
                    prio = "Haute" if s.priority == 1 else "Moyenne" if s.priority == 2 else "Basse"
                    print(f"\n   {i}. [{s.type.upper()}] {s.description}")
                    print(f"      Priorité: {prio}")
                    if s.explanation:
                        print("\n      Explication:")
                        print(f"      {s.explanation}")

                    # optionnel: afficher patch / code corrigé (pas trop long)
                    if s.type in ("ai_patch", "ai_fixed_code"):
                        print("\n      Proposition:")
                        preview = s.suggested_code
                        if preview and len(preview) > 1200:
                            preview = preview[:1200] + "\n... (tronqué)"
                        print(preview)

        print("=" * 80)

    def generate_summary_report(self, results: List[AnalysisResult]) -> str:
        """Rapport global pour plusieurs fichiers (utile pour analyse de dossier)."""

        report = []
        report.append("=" * 80)
        report.append(" RAPPORT D'ANALYSE DE CODE (GLOBAL)")
        report.append("=" * 80)
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Fichiers analysés: {len(results)}\n")

        ok_results = [r for r in results if r.success]
        total_problems = sum(r.metrics.total_problems for r in ok_results)
        total_functions = sum(r.metrics.total_functions for r in ok_results)
        total_classes = sum(r.metrics.total_classes for r in ok_results)

        report.append(" STATISTIQUES")
        report.append(f"  Fonctions: {total_functions}")
        report.append(f"  Classes: {total_classes}")
        report.append(f"  Problèmes: {total_problems}\n")

        report.append(" DÉTAILS PAR FICHIER")
        for r in results:
            if not r.success:
                report.append(f" {r.file_path} (échec)")
                continue
            status = "" if not r.has_problems() else ""
            report.append(f"{status} {r.file_path}")
            report.append(f"   Langage: {r.language} | Problèmes: {r.metrics.total_problems}")

        report.append("=" * 80)
        return "\n".join(report)

