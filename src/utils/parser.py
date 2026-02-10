"""
Utilitaires pour le parsing de code avec Tree-sitter.

détection d'erreurs de parsing (ERROR nodes) + root.has_error
"""

#Tree-sitter nécessite une grammaire spécifique par langage
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

import tree_sitter_python as tspython
import tree_sitter_javascript as tsjs
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser


class CodeParser:
    """
    Parser de code multi-langages utilisant Tree-sitter.
    """

    LANGUAGE_MAP = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java'
    } #Mapper les extensions de fichiers vers les langages supportés.

    def __init__(self):
        self.parsers: Dict[str, Any] = {}
        self._load_languages()

    def _load_languages(self):
        try:
           
            try:
                python_lang = Language(tspython.language(), 'python')
                api_type = "NEW"
                print(" Utilisation de l'API tree-sitter moderne")
            except (TypeError, AttributeError):
                python_lang = Language(tspython.language())
                api_type = "OLD"
                print(" Utilisation de l'API tree-sitter ancienne")

            # Python
            python_parser = Parser()
            python_parser.set_language(python_lang)
            self.parsers['python'] = (python_lang, python_parser) # Grammaire,  Moteur de parsing

            # JS/TS
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

            java_parser = Parser()  # Crée une instance vide
            java_parser.set_language(java_lang) # Associe la grammaire Java
            self.parsers['java'] = (java_lang, java_parser)

            print(f" Langages chargés : {list(self.parsers.keys())}")

        except Exception as e:
            print(f"  Erreur lors du chargement des langages : {e}")
            raise


#detect_language("app.py")       # → 'python'
    def detect_language(self, file_path: str) -> Optional[str]:
        ext = Path(file_path).suffix.lower()
        return self.LANGUAGE_MAP.get(ext)
    
  #Vérifier que le fichier existe  
    def parse_file(self, file_path: str) -> Optional[Any]:
        if not os.path.exists(file_path):
            print(f" Fichier non trouvé : {file_path}")
            return None
   #Détecter le langage via l'extension
        language = self.detect_language(file_path)
        if not language:
            print(f" Langage non supporté pour : {file_path}")
            return None

        try:
            #Lire le contenu du fichier
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            return self.parse_code(code, language)
        except Exception as e:
            print(f" Erreur lors de la lecture du fichier : {e}")
            return None
        
        

    def parse_code(self, code: str, language: str) -> Optional[Any]:
          #VÉRIFICATION DU LANGAGE
        if language not in self.parsers:
            print(f" Langage '{language}' non supporté")
            return None

        try:
             #RÉCUPÉRATION DU PARSER
            _, parser = self.parsers[language]
             
            #PARSING
            tree = parser.parse(bytes(code, "utf8"))  # Convertit str → bytes
            return tree.root_node # Racine de l'AST
        except Exception as e:
            print(f" Erreur lors du parsing : {e}")
            return None

    
   #NOUVEAU : erreurs Tree-sitter
    def get_parse_errors(self, ast_node: Any, code: str, max_errors: int = 20) -> List[dict]:
        """
        Retourne une liste d'erreurs de parsing Tree-sitter (nœuds ERROR).
        """
        errors: List[dict] = []
        if ast_node is None:
            return errors

        def traverse(node):
            nonlocal errors
            if len(errors) >= max_errors:
                return
            if node.type == "ERROR":
                 # Calcul des positions (Tree-sitter démarre à 0)
                start_line, start_col = node.start_point[0] + 1, node.start_point[1] + 1
                end_line, end_col = node.end_point[0] + 1, node.end_point[1] + 1
                # Extraction du code erroné
                snippet = code[node.start_byte:node.end_byte]
                
                
                errors.append({
                    "message": "Erreur de syntaxe détectée par Tree-sitter",
                    "location": f"l.{start_line}:{start_col} - l.{end_line}:{end_col}",
                    "snippet": snippet[:200]
                })
              
               # PARCOURS RÉCURSIF 
            for child in node.children:
                traverse(child)

        traverse(ast_node)
        return errors
    
#Tree-sitter marque has_error=True si l'AST contient des erreurs
    def has_parse_error(self, ast_node: Any) -> bool:
        return bool(ast_node) and getattr(ast_node, "has_error", False)

    # -------------------------
   
   
    def extract_functions(self, ast_node: Any) -> list:
        functions = []

        def traverse(node):
            function_types = [
                'function_definition',
                'function_declaration',
                'method_definition',
                'arrow_function',
                'function_expression',
                'method_declaration',
            ]
             #DÉTECTION 
            if node.type in function_types:
                functions.append(node)
            #PARCOURS RÉCURSIF  
            for child in node.children:
                traverse(child)

        traverse(ast_node)
        return functions
    
    
# Trouver les Classes
    def extract_classes(self, ast_node: Any) -> list:
        classes = []

        def traverse(node):
            class_types = ['class_definition', 'class_declaration']
            if node.type in class_types:
                classes.append(node)
            for child in node.children:
                traverse(child)

        traverse(ast_node)
        return classes


#Trouver les Imports
    def extract_imports(self, ast_node: Any) -> list:
        imports = []

        def traverse(node):
            import_types = [
                'import_statement',
                'import_from_statement',
                'import_declaration',
                'import',
            ]
            if node.type in import_types:
                imports.append(node)
            for child in node.children:
                traverse(child)

        traverse(ast_node)
        return imports
    
    
#Extraire le Code d'un Node
    def get_node_text(self, node: Any, code: str) -> str:
        return code[node.start_byte:node.end_byte]

    def calculate_complexity(self, node: Any) -> int:
        complexity = 1
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
    
#Compter les Lignes
    def count_lines(self, node: Any, code: str) -> int:
        text = self.get_node_text(node, code)
        return len([line for line in text.split('\n') if line.strip()])
    
    
#Profondeur d'Imbrication
    def calculate_nesting_depth(self, node: Any) -> int:
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
    
#Détection de Duplication
    def find_duplicate_code(self, ast_node: Any, code: str, min_length: int = 10) -> list:
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
