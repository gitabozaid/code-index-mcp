"""Code chunking using Tree-sitter for semantic code splitting."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

# Tree-sitter imports
try:
    import tree_sitter_python as tspython
    import tree_sitter_javascript as tsjavascript
    import tree_sitter_typescript as tstypescript
    import tree_sitter_go as tsgo
    import tree_sitter_rust as tsrust
    import tree_sitter_java as tsjava
    import tree_sitter_cpp as tscpp
    from tree_sitter import Language, Parser, Query
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logging.warning("Tree-sitter languages not available. Code chunking will use fallback method.")

logger = logging.getLogger(__name__)


class CodeChunker:
    """
    Chunks code files into semantic units using Tree-sitter parsing.

    Strategy:
    - Functions/methods: Each function is a separate chunk
    - Classes: Each class is a separate chunk (including its methods)
    - Top-level statements: Grouped into chunks (max 512 tokens)
    - Comments: Preserved with their associated code
    """

    # Language configurations
    LANGUAGES = {}
    if TREE_SITTER_AVAILABLE:
        LANGUAGES = {
            'python': Language(tspython.language()),
            'javascript': Language(tsjavascript.language()),
            'typescript': Language(tstypescript.language()),
            'go': Language(tsgo.language()),
            'rust': Language(tsrust.language()),
            'java': Language(tsjava.language()),
            'cpp': Language(tscpp.language()),
        }

    # Tree-sitter queries for extracting functions/classes
    QUERIES = {
        'python': """
            (function_definition
                name: (identifier) @func.name
                body: (block) @func.body) @function

            (class_definition
                name: (identifier) @class.name
                body: (block) @class.body) @class
        """,
        'javascript': """
            (function_declaration
                name: (identifier) @func.name
                body: (statement_block) @func.body) @function

            (class_declaration
                name: (identifier) @class.name
                body: (class_body) @class.body) @class
        """,
        'typescript': """
            (function_declaration
                name: (identifier) @func.name
                body: (statement_block) @func.body) @function

            (class_declaration
                name: (type_identifier) @class.name
                body: (class_body) @class.body) @class
        """,
        'go': """
            (function_declaration
                name: (identifier) @func.name
                body: (block) @func.body) @function

            (method_declaration
                name: (field_identifier) @method.name
                body: (block) @method.body) @method
        """,
        'rust': """
            (function_item
                name: (identifier) @func.name
                body: (block) @func.body) @function

            (impl_item
                type: (type_identifier) @impl.name) @impl
        """,
        'java': """
            (method_declaration
                name: (identifier) @method.name
                body: (block) @method.body) @method

            (class_declaration
                name: (identifier) @class.name
                body: (class_body) @class.body) @class
        """,
        'cpp': """
            (function_definition
                declarator: (function_declarator
                    declarator: (identifier) @func.name)) @function

            (class_specifier
                name: (type_identifier) @class.name
                body: (field_declaration_list) @class.body) @class
        """,
    }

    MAX_CHUNK_SIZE = 512  # tokens (approximate using character count * 0.25)

    def __init__(self):
        """Initialize chunker with language parsers."""
        self.parsers = {}
        self.queries = {}

        if not TREE_SITTER_AVAILABLE:
            logger.warning("Tree-sitter not available. Using fallback chunking.")
            return

        for lang, language in self.LANGUAGES.items():
            parser = Parser(language)
            self.parsers[lang] = parser

            if lang in self.QUERIES:
                try:
                    self.queries[lang] = Query(language, self.QUERIES[lang])
                except Exception as e:
                    logger.warning(f"Failed to create query for {lang}: {e}")

    def detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension."""
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.go': 'go',
            '.rs': 'rust',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cc': 'cpp',
            '.cxx': 'cpp',
            '.h': 'cpp',
            '.hpp': 'cpp',
            '.hxx': 'cpp',
        }

        path = Path(file_path)
        return ext_map.get(path.suffix)

    def chunk_file(
        self,
        file_path: str,
        content: str,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk a code file into semantic units.

        Args:
            file_path: Path to the file
            content: File content as string
            language: Programming language (auto-detected if None)

        Returns:
            List of chunk dictionaries with metadata
        """
        if language is None:
            language = self.detect_language(file_path)

        if not TREE_SITTER_AVAILABLE or not language or language not in self.parsers:
            # Fallback: return entire file as single chunk
            return [self._create_chunk(
                file_path=file_path,
                content=content,
                start_line=1,
                end_line=content.count('\n') + 1,
                chunk_type='file',
                symbol_name=Path(file_path).name
            )]

        parser = self.parsers[language]
        try:
            tree = parser.parse(bytes(content, 'utf8'))
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {e}. Using fallback.")
            return [self._create_chunk(
                file_path=file_path,
                content=content,
                start_line=1,
                end_line=content.count('\n') + 1,
                chunk_type='file',
                symbol_name=Path(file_path).name
            )]

        chunks = []

        # Extract functions and classes using Tree-sitter queries
        if language in self.queries:
            query = self.queries[language]
            try:
                captures = query.captures(tree.root_node)

                for node, capture_name in captures:
                    chunk = self._node_to_chunk(
                        file_path=file_path,
                        content=content,
                        node=node,
                        chunk_type=capture_name.split('.')[0]  # 'function' or 'class'
                    )
                    chunks.append(chunk)
            except Exception as e:
                logger.warning(f"Query capture failed for {file_path}: {e}")

        # If no chunks extracted, fallback to entire file
        if not chunks:
            chunks.append(self._create_chunk(
                file_path=file_path,
                content=content,
                start_line=1,
                end_line=content.count('\n') + 1,
                chunk_type='file',
                symbol_name=Path(file_path).name
            ))

        logger.debug(f"Chunked {file_path}: {len(chunks)} chunks")
        return chunks

    def _node_to_chunk(
        self,
        file_path: str,
        content: str,
        node,
        chunk_type: str
    ) -> Dict[str, Any]:
        """Convert Tree-sitter node to chunk dictionary."""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # Extract node text
        node_text = content[node.start_byte:node.end_byte]

        # Extract symbol name (function/class name)
        symbol_name = "unknown"
        for child in node.children:
            if child.type in ('identifier', 'type_identifier', 'field_identifier'):
                symbol_name = content[child.start_byte:child.end_byte]
                break

        return self._create_chunk(
            file_path=file_path,
            content=node_text,
            start_line=start_line,
            end_line=end_line,
            chunk_type=chunk_type,
            symbol_name=symbol_name
        )

    def _create_chunk(
        self,
        file_path: str,
        content: str,
        start_line: int,
        end_line: int,
        chunk_type: str,
        symbol_name: str
    ) -> Dict[str, Any]:
        """Create chunk metadata dictionary."""
        # Generate unique chunk ID
        chunk_id = hashlib.md5(
            f"{file_path}:{start_line}:{end_line}".encode()
        ).hexdigest()[:16]

        return {
            'id': chunk_id,
            'file_path': file_path,
            'content': content,
            'start_line': start_line,
            'end_line': end_line,
            'line_count': end_line - start_line + 1,
            'chunk_type': chunk_type,  # 'function', 'class', 'file'
            'symbol_name': symbol_name,
            'file_hash': hashlib.md5(content.encode()).hexdigest(),
        }

    def chunk_text(self, text: str, max_size: int = 512) -> List[str]:
        """
        Simple text chunking for non-code files.

        Args:
            text: Input text
            max_size: Maximum chunk size in characters

        Returns:
            List of text chunks
        """
        if len(text) <= max_size:
            return [text]

        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_size = 0

        for line in lines:
            line_size = len(line) + 1  # +1 for newline
            if current_size + line_size > max_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks
