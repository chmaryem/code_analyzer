"""
Utilitaires pour le parsing de code avec Tree-sitter.
Compatible avec toutes les versions de tree-sitter (ancienne et nouvelle API).
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjs
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser


class CodeParser:
    """
    Parser de code multi-langages utilisant Tree-sitter.
    Compatible avec l'ancienne et la nouvelle API.
    """

    # Mapping des langages vers les modules de grammaires Tree-sitter
    LANGUAGE_MAP = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.cc': 'cpp',
        '.cxx': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
    }

    def __init__(self):
        """
        Initialise le parser pour tous les langages supportés.
        """
        self.parsers = {}
        self._load_languages()
        

    def _load_languages(self):
        """
        Charge les langages Tree-sitter supportés.
        Détecte automatiquement l'API (ancienne ou nouvelle).
        """
        try:
            # Essayer d'abord la nouvelle API (tree-sitter >= 0.21.0)
            try:
                python_lang = Language(tspython.language(), 'python')
                api_type = "NEW"
                print("🔧 Utilisation de l'API tree-sitter moderne")
            except (TypeError, AttributeError):
                # Ancienne API (tree-sitter < 0.21.0)
                python_lang = Language(tspython.language())
                api_type = "OLD"
                print("🔧 Utilisation de l'API tree-sitter ancienne")
            
            # Python
            python_parser = Parser()
            python_parser.set_language(python_lang)
            self.parsers['python'] = (python_lang, python_parser)
            
            # JavaScript/TypeScript
            if api_type == "NEW":
                js_lang = Language(tsjs.language(), 'javascript')
            else:
                js_lang = Language(tsjs.language())
            
            js_parser = Parser()
            js_parser.set_language(js_lang)
            self.parsers['javascript'] = (js_lang, js_parser)
            self.parsers['typescript'] = (js_lang, js_parser)
            
            # Java
            if api_type == "NEW":
                java_lang = Language(tsjava.language(), 'java')
            else:
                java_lang = Language(tsjava.language())
            
            java_parser = Parser()
            java_parser.set_language(java_lang)
            self.parsers['java'] = (java_lang, java_parser)
            
            print(f"✅ Langages chargés : {list(self.parsers.keys())}")
            
        except Exception as e:
            print(f"⚠️  Erreur lors du chargement des langages : {e}")
            raise
    

    def detect_language(self, file_path: str) -> Optional[str]:
        """
        Détecte le langage de programmation à partir de l'extension du fichier.

        Args:
            file_path: Chemin du fichier

        Returns:
            Langage détecté ou None
        """
        ext = Path(file_path).suffix.lower()
        return self.LANGUAGE_MAP.get(ext)

    def parse_file(self, file_path: str) -> Optional[Any]:
        """
        Parse un fichier et retourne l'AST
        
        Args:
            file_path: Chemin du fichier à parser
            
        Returns:
            Noeud racine de l'AST ou None en cas d'erreur
        """
        if not os.path.exists(file_path):
            print(f"❌ Fichier non trouvé : {file_path}")
            return None
        
        language = self.detect_language(file_path)
        if not language:
            print(f"❌ Langage non supporté pour : {file_path}")
            return None
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            
            return self.parse_code(code, language)
            
        except Exception as e:
            print(f"❌ Erreur lors de la lecture du fichier : {e}")
            return None

    def parse_code(self, code: str, language: str) -> Optional[Any]:
        """
        Parse une chaîne de code et retourne l'AST
        
        Args:
            code: Code source à parser
            language: Langage de programmation
            
        Returns:
            Noeud racine de l'AST ou None en cas d'erreur
        """
        if language not in self.parsers:
            print(f"❌ Langage '{language}' non supporté")
            return None
        
        try:
            _, parser = self.parsers[language]
            tree = parser.parse(bytes(code, "utf8"))
            return tree.root_node
            
        except Exception as e:
            print(f"❌ Erreur lors du parsing : {e}")
            return None
    
    def extract_functions(self, ast_node: Any) -> list:
        """
        Extrait toutes les fonctions de l'AST
        
        Args:
            ast_node: Noeud racine de l'AST
            
        Returns:
            Liste des nœuds de fonctions
        """
        functions = []
        
        def traverse(node):
            # Types de nœuds pour fonctions selon les langages
            function_types = [
                'function_definition',      # Python
                'function_declaration',     # JavaScript/Java
                'method_definition',        # JavaScript classes
                'arrow_function',          # JavaScript
                'function_expression',     # JavaScript
                'method_declaration',      # Java
            ]
            
            if node.type in function_types:
                functions.append(node)
            for child in node.children:
                traverse(child)
        
        traverse(ast_node)
        return functions
    
    def extract_classes(self, ast_node: Any) -> list:
        """
        Extrait toutes les classes de l'AST
        
        Args:
            ast_node: Noeud racine de l'AST
            
        Returns:
            Liste des nœuds de classes
        """
        classes = []
        
        def traverse(node):
            class_types = [
                'class_definition',    # Python
                'class_declaration',   # JavaScript/Java
            ]
            
            if node.type in class_types:
                classes.append(node)
            for child in node.children:
                traverse(child)
        
        traverse(ast_node)
        return classes
    
    def extract_imports(self, ast_node: Any) -> list:
        """
        Extrait tous les imports de l'AST
        
        Args:
            ast_node: Noeud racine de l'AST
            
        Returns:
            Liste des imports trouvés
        """
        imports = []
        
        def traverse(node):
            import_types = [
                'import_statement',        # Python
                'import_from_statement',   # Python
                'import_declaration',      # JavaScript
                'import',                  # Java
            ]
            
            if node.type in import_types:
                imports.append(node)
            for child in node.children:
                traverse(child)
        
        traverse(ast_node)
        return imports
    
    def get_node_text(self, node: Any, code: str) -> str:
        """
        Récupère le texte d'un nœud depuis le code source
        
        Args:
            node: Noeud de l'AST
            code: Code source original
            
        Returns:
            Texte du nœud
        """
        return code[node.start_byte:node.end_byte]
    
    def calculate_complexity(self, node: Any) -> int:
        """
        Calcule la complexité cyclomatique d'un nœud
        
        Args:
            node: Noeud de l'AST
            
        Returns:
            Score de complexité
        """
        complexity = 1
        
        # Instructions conditionnelles et boucles augmentent la complexité
        control_structures = [
            'if_statement', 'elif_clause', 'else_clause',
            'for_statement', 'while_statement',
            'try_statement', 'except_clause',
            'switch_statement', 'case_statement',
            'conditional_expression',
            'ternary_expression',
            'and', 'or',
        ]
        
        def traverse(n):
            nonlocal complexity
            if n.type in control_structures:
                complexity += 1
            for child in n.children:
                traverse(child)
        
        traverse(node)
        return complexity
    
    def count_lines(self, node: Any, code: str) -> int:
        """
        Compte le nombre de lignes d'un nœud
        
        Args:
            node: Noeud de l'AST
            code: Code source original
            
        Returns:
            Nombre de lignes
        """
        text = self.get_node_text(node, code)
        return len([line for line in text.split('\n') if line.strip()])
    
    def calculate_nesting_depth(self, node: Any) -> int:
        """
        Calcule la profondeur d'imbrication d'un nœud
        
        Args:
            node: Noeud de l'AST
            
        Returns:
            Profondeur maximale d'imbrication
        """
        def get_depth(n, current_depth=0):
            max_depth = current_depth
            nesting_types = [
                'block', 'function_definition', 'class_definition',
                'if_statement', 'for_statement', 'while_statement',
                'try_statement', 'with_statement', 'function_declaration',
                'method_definition', 'arrow_function',
            ]
            
            for child in n.children:
                if child.type in nesting_types:
                    child_depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)
            return max_depth
        
        return get_depth(node)
    
    def find_duplicate_code(self, ast_node: Any, code: str, min_length: int = 10) -> list:
        """
        Détecte le code dupliqué
        
        Args:
            ast_node: Noeud racine de l'AST
            code: Code source
            min_length: Longueur minimale pour considérer une duplication
            
        Returns:
            Liste des duplications trouvées
        """
        code_blocks = {}
        duplicates = []
        
        def traverse(node):
            if node.type in ['function_definition', 'if_statement', 'for_statement', 'while_statement']:
                block_text = self.get_node_text(node, code).strip()
                if len(block_text) >= min_length:
                    if block_text in code_blocks:
                        duplicates.append({
                            'original': code_blocks[block_text],
                            'duplicate': node,
                            'text': block_text[:100] + '...' if len(block_text) > 100 else block_text
                        })
                    else:
                        code_blocks[block_text] = node
            
            for child in node.children:
                traverse(child)
        
        traverse(ast_node)
        return duplicates