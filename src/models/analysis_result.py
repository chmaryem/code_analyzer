"""
Modèles de données pour les résultats d'analyse de code.
Ces modèles définissent la structure des données retournées par l'agent.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class ProblemSeverity(str, Enum):
    """Niveaux de gravité des problèmes"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ProblemCategory(str, Enum):
    """Catégories de problèmes"""
    COMPLEXITY = "complexity"
    LENGTH = "length"
    NESTING = "nesting" # Imbrication excessive
    NAMING = "naming"  # Nommage non conforme
    DOCUMENTATION = "documentation"
    SECURITY = "security"  # Faille de sécurité
    PERFORMANCE = "performance" # Problème de performance
    STYLE = "style"


class CodeSuggestion(BaseModel):
    """Suggestion de correction pour un problème"""
    type: str = Field(..., description="Type de suggestion (refactor, rename, extract, etc.)")
    description: str = Field(..., description="Description de la suggestion")
    original_code: str = Field(..., description="Code original à modifier")
    suggested_code: str = Field(..., description="Code suggéré après correction")
    explanation: str = Field(..., description="Explication détaillée du changement")
    priority: int = Field(default=1, description="Priorité de la suggestion (1=haut, 3=bas)")
    
    def __str__(self):
        return f"{self.type}: {self.description}"


class CodeProblem(BaseModel):
    """Un problème détecté dans le code"""
    category: ProblemCategory = Field(..., description="Catégorie du problème")
    severity: ProblemSeverity = Field(default=ProblemSeverity.WARNING, description="Gravité du problème")
    message: str = Field(..., description="Description du problème")
    location: Optional[str] = Field(None, description="Emplacement dans le code (ligne/colonne)")
    details: Optional[dict] = Field(default_factory=dict, description="Détails supplémentaires")
    suggestions: List[CodeSuggestion] = Field(default_factory=list, description="Suggestions de correction")
    
    def add_suggestion(self, suggestion: CodeSuggestion):
        """Ajoute une suggestion de correction"""
        self.suggestions.append(suggestion)
    
    def has_suggestions(self) -> bool:
        """Vérifie si des suggestions sont disponibles"""
        return len(self.suggestions) > 0
    
    def __str__(self):
        severity_emoji = {
            ProblemSeverity.INFO: "info:",
            ProblemSeverity.WARNING: "warning:",
            ProblemSeverity.ERROR: " error:",
            ProblemSeverity.CRITICAL: " critical:"
        }
        emoji = severity_emoji.get(self.severity, " ")
        return f"{emoji} {self.message} ({self.category.value})"


class FunctionInfo(BaseModel):
    """Informations sur une fonction"""
    name: str = Field(..., description="Nom de la fonction")
    start_line: int = Field(..., description="Ligne de début")
    end_line: int = Field(..., description="Ligne de fin")
    line_count: int = Field(..., description="Nombre de lignes")
    complexity: int = Field(..., description="Complexité cyclomatique")
    nesting_depth: int = Field(..., description="Profondeur d'imbrication")
    problems: List[CodeProblem] = Field(default_factory=list, description="Problèmes détectés")
    
    def add_problem(self, problem: CodeProblem):
        """Ajoute un problème à la fonction"""
        self.problems.append(problem)
    
    def has_problems(self) -> bool:
        """Vérifie si des problèmes ont été détectés"""
        return len(self.problems) > 0
    
    def __str__(self):
        status = "✅" if not self.has_problems() else "⚠️"
        return f"{status} {self.name} (l. {self.start_line}-{self.end_line}, {self.line_count} lignes, complexité: {self.complexity})"


class ClassInfo(BaseModel):
    """Informations sur une classe"""
    name: str = Field(..., description="Nom de la classe")
    start_line: int = Field(..., description="Ligne de début")
    end_line: int = Field(..., description="Ligne de fin")
    method_count: int = Field(..., description="Nombre de méthodes")
    problems: List[CodeProblem] = Field(default_factory=list, description="Problèmes détectés")
    
    def add_problem(self, problem: CodeProblem):
        """Ajoute un problème à la classe"""
        self.problems.append(problem)
    
    def has_problems(self) -> bool:
        """Vérifie si des problèmes ont été détectés"""
        return len(self.problems) > 0
    
    def __str__(self):
        status = "✅" if not self.has_problems() else "⚠️"
        return f"{status} {self.name} (l. {self.start_line}-{self.end_line}, {self.method_count} méthodes)"


class AnalysisMetrics(BaseModel):
    """Métriques globales de l'analyse"""
    total_functions: int = Field(default=0, description="Nombre total de fonctions")
    total_classes: int = Field(default=0, description="Nombre total de classes")
    total_problems: int = Field(default=0, description="Nombre total de problèmes")
    critical_problems: int = Field(default=0, description="Nombre de problèmes critiques")
    error_problems: int = Field(default=0, description="Nombre de problèmes d'erreur")
    warning_problems: int = Field(default=0, description="Nombre d'avertissements")
    info_problems: int = Field(default=0, description="Nombre d'informations")
    functions_with_problems: int = Field(default=0, description="Nombre de fonctions avec problèmes")
    classes_with_problems: int = Field(default=0, description="Nombre de classes avec problèmes")
    
    def update_from_results(self, functions: List[FunctionInfo], classes: List[ClassInfo]):
        """Met à jour les métriques à partir des résultats"""
        self.total_functions = len(functions)
        self.total_classes = len(classes)
        
        for func in functions:
            if func.has_problems():
                self.functions_with_problems += 1
                for problem in func.problems:
                    self.total_problems += 1
                    if problem.severity == ProblemSeverity.CRITICAL:
                        self.critical_problems += 1
                    elif problem.severity == ProblemSeverity.ERROR:
                        self.error_problems += 1
                    elif problem.severity == ProblemSeverity.WARNING:
                        self.warning_problems += 1
                    elif problem.severity == ProblemSeverity.INFO:
                        self.info_problems += 1
        
        for cls in classes:
            if cls.has_problems():
                self.classes_with_problems += 1
                for problem in cls.problems:
                    self.total_problems += 1


class AnalysisResult(BaseModel):
    """Résultat complet de l'analyse de code"""
    file_path: str = Field(..., description="Chemin du fichier analysé")
    language: str = Field(..., description="Langage de programmation")
    success: bool = Field(..., description="Si l'analyse a réussi")
    timestamp: str = Field(default="", description="Timestamp de l'analyse")
    functions: List[FunctionInfo] = Field(default_factory=list, description="Fonctions analysées")
    classes: List[ClassInfo] = Field(default_factory=list, description="Classes analysées")
    imports: List[str] = Field(default_factory=list, description="Imports trouvés")
    metrics: AnalysisMetrics = Field(default_factory=AnalysisMetrics, description="Métriques globales")
    
    def calculate_metrics(self):
        """Calcule les métriques globales"""
        self.metrics.update_from_results(self.functions, self.classes)
    
    def has_problems(self) -> bool:
        """Vérifie si des problèmes ont été détectés"""
        return self.metrics.total_problems > 0
    
    def get_problems_by_severity(self, severity: ProblemSeverity) -> List[CodeProblem]:
        """Récupère tous les problèmes d'une gravité donnée"""
        problems = []
        for func in self.functions:
            for problem in func.problems:
                if problem.severity == severity:
                    problems.append(problem)
        for cls in self.classes:
            for problem in cls.problems:
                if problem.severity == severity:
                    problems.append(problem)
        return problems
    
    def get_all_problems(self) -> List[CodeProblem]:
        """Récupère tous les problèmes détectés"""
        problems = []
        for func in self.functions:
            problems.extend(func.problems)
        for cls in self.classes:
            problems.extend(cls.problems)
        return problems
    
    def get_all_suggestions(self) -> List[CodeSuggestion]:
        """Récupère toutes les suggestions de correction"""
        suggestions = []
        for func in self.functions:
            for problem in func.problems:
                suggestions.extend(problem.suggestions)
        for cls in self.classes:
            for problem in cls.problems:
                suggestions.extend(problem.suggestions)
        return suggestions
    
    def __str__(self):
        status = "✅" if self.success else "❌"
        problems_status = "🎉" if not self.has_problems() else "⚠️"
        return f"{status} {self.file_path} ({self.language}) - {self.metrics.total_problems} problèmes {problems_status}"