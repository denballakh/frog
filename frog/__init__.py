#!/usr/bin/env python3
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import assert_never, cast
from enum import Enum
from dataclasses import dataclass, field

from .sb import StringBuilder
from .logs import error, notimplemented, warn, info, trace, unreachable, typecheck_has_a_bug
from .types import (
    INTRINSIC_TO_INTRINSIC_TYPE,
    KW_TO_KWT,
    missing,
    unused,
    typechecking,
    Loc,
    loc_unknown,
    TokenType,
    Token,
    KeywordType,
    InstructionType,
    Instruction,
    ValueCls,
    ValueClsType,
    StackEntry,
    Stack,
    Proc,
    ProcRef,
    Module,
    Contract,
    IR,
    Error,
    IntrinsicType,
    pp,
)


@dataclass(frozen=True)
class MemoryPtr:
    buf: bytearray
    offset: int = 0


MEMORY_ACCESS_BITS = {
    IntrinsicType.READ_I8: (8, True),
    IntrinsicType.READ_I16: (16, True),
    IntrinsicType.READ_I32: (32, True),
    IntrinsicType.READ_I64: (64, True),
    IntrinsicType.READ_U8: (8, False),
    IntrinsicType.READ_U16: (16, False),
    IntrinsicType.READ_U32: (32, False),
    IntrinsicType.READ_U64: (64, False),
    IntrinsicType.WRITE_I8: (8, True),
    IntrinsicType.WRITE_I16: (16, True),
    IntrinsicType.WRITE_I32: (32, True),
    IntrinsicType.WRITE_I64: (64, True),
    IntrinsicType.WRITE_U8: (8, False),
    IntrinsicType.WRITE_U16: (16, False),
    IntrinsicType.WRITE_U32: (32, False),
    IntrinsicType.WRITE_U64: (64, False),
}


def expect_enum_size(e_cls: type[Enum], expected_size: int) -> None:
    if len(e_cls) != expected_size:
        raise Error(f'{e_cls.__name__} has {len(e_cls)} members, expected {expected_size}')


def _tokenize(text: str, filename: str = '<?>') -> Iterable[Token]:
    FIRST_LINE = 1
    FIRST_COL = 1

    def is_int(s: str) -> bool:
        return set(s) <= set('0123456789')

    line_no = FIRST_LINE
    col_no = FIRST_COL

    loc_start = Loc(filename, line_no, col_no)
    i_start = 0

    text += '\n'

    i = 0
    while i < len(text):
        c = text[i]

        col_no += 1
        if c == '\n':
            line_no += 1
            col_no = FIRST_COL

        if c == '\'':
            chars = ''
            while True:
                i += 1
                if i >= len(text):
                    error(loc_start, f'unterminated character literal')
                if text[i] == '\n':
                    error(loc_start, f'unterminated character literal')
                if text[i] == '\'':
                    i += 1
                    break
                chars += text[i]
            if len(chars) != 1:
                error(loc_start, f'invalid character literal')
            yield Token(TokenType.CHAR, chars, loc_start)

            loc_start = Loc(filename, line_no, col_no)
            i_start = i + 1
            continue

        if c == '"':
            chars = ''
            while True:
                i += 1
                if i >= len(text):
                    error(loc_start, f'unterminated string literal')
                if text[i] == '\n':
                    error(loc_start, f'unterminated string literal')
                if text[i] == '"':
                    i += 1
                    break
                chars += text[i]
            yield Token(TokenType.STR, chars, loc_start)

            loc_start = Loc(filename, line_no, col_no)
            i_start = i + 1
            continue

        if c in {' ', '\t', '\n'}:
            chunk = text[i_start:i]

            if not chunk:
                pass
            elif chunk == '//':
                while i < len(text) and text[i] != '\n':
                    i += 1
                i -= 1
            elif chunk == 'true' or chunk == 'false':
                yield Token(TokenType.BOOL, chunk == 'true', loc_start)
            elif is_int(chunk):
                yield Token(TokenType.INT, int(chunk), loc_start)
            else:
                if chunk in KW_TO_KWT:
                    yield Token(TokenType.KEYWORD, KW_TO_KWT[chunk], loc_start)
                else:
                    yield Token(TokenType.WORD, chunk, loc_start)

            loc_start = Loc(filename, line_no, col_no)
            i_start = i + 1

        i += 1


def tokenize(text: str, filename: str = '<?>') -> list[Token]:
    toks = [*_tokenize(text, filename)]
    trace(
        loc_unknown,
        f'file: {filename}',
        Tokens=toks,
    )
    return toks


@dataclass
class Macro:
    name: str
    tok: Token
    body: list[Token]


class SymbolType(Enum):
    PROC = 'proc'
    MACRO = 'macro'


@dataclass(frozen=True)
class Symbol:
    type: SymbolType
    module: str
    name: str


@dataclass
class ImportItem:
    name: str
    alias: str
    tok: Token


@dataclass
class ImportDecl:
    path: str
    tok: Token
    items: list[ImportItem]


@dataclass
class ModuleSource:
    id: str
    path: str
    canonical_path: Path | None
    toks: list[Token]
    imports: list[ImportDecl]
    body_toks: list[Token]
    macros: dict[str, Macro] = field(default_factory=dict)
    proc_toks: dict[str, Token] = field(default_factory=dict)
    imported_symbols: dict[str, Symbol] = field(default_factory=dict)
    scope: dict[str, Symbol] = field(default_factory=dict)


def _display_path(path: Path) -> str:
    path = path.resolve(strict=False)
    cwd = Path.cwd().resolve(strict=False)
    try:
        return str(path.relative_to(cwd))
    except ValueError:
        return str(path)


def _normalize_import_path(root_dir: Path, path: str) -> Path:
    return (root_dir / path).resolve(strict=False)


def _is_word(tok: Token, value: str) -> bool:
    return tok.type == TokenType.WORD and tok.value == value


def _is_keyword(tok: Token, value: KeywordType) -> bool:
    return tok.type == TokenType.KEYWORD and tok.value == value


def _parse_import_item(toks: list[Token], i: int) -> tuple[ImportItem, int]:
    if i >= len(toks):
        error(toks[-1] if toks else None, 'expected imported name after import')

    name_tok = toks[i]
    if name_tok.type != TokenType.WORD:
        error(name_tok, f'expected imported name after import, got {name_tok.type}')
    if name_tok.value == '*':
        error(name_tok, 'wildcard imports are not supported')
    if ',' in name_tok.value:
        error(name_tok, 'commas are not valid in import lists')
    if name_tok.value in {'(', ')'}:
        error(name_tok, f'expected imported name after import, got {name_tok.value}')

    name = name_tok.value
    alias = name
    i += 1
    if i < len(toks) and _is_keyword(toks[i], KeywordType.AS):
        i += 1
        if i >= len(toks):
            error(name_tok, f'expected an alias after as')
        alias_tok = toks[i]
        if alias_tok.type != TokenType.WORD:
            error(alias_tok, f'expected an alias after as, got {alias_tok.type}')
        if alias_tok.value == '*' or ',' in alias_tok.value or alias_tok.value in {'(', ')'}:
            error(alias_tok, f'invalid import alias {alias_tok.value}')
        alias = alias_tok.value
        i += 1

    return ImportItem(name=name, alias=alias, tok=name_tok), i


def _parse_import_decl(toks: list[Token], i: int) -> tuple[ImportDecl, int]:
    tok_from = toks[i]
    i += 1

    if i >= len(toks) or toks[i].type != TokenType.STR:
        error(tok_from, 'expected a string path after from')
    path_tok = toks[i]
    path = cast(str, path_tok.value)
    i += 1

    if i >= len(toks) or not _is_keyword(toks[i], KeywordType.IMPORT):
        error(path_tok, 'expected import after module path')
    i += 1

    items: list[ImportItem] = []
    if i >= len(toks):
        error(path_tok, 'expected imported name after import')

    if _is_word(toks[i], '('):
        i += 1
        while i < len(toks) and not _is_word(toks[i], ')'):
            if _is_word(toks[i], ','):
                error(toks[i], 'commas are not valid in import lists')
            item, i = _parse_import_item(toks, i)
            items.append(item)
        if i >= len(toks):
            error(path_tok, 'expected ) after import list')
        i += 1
        if not items:
            error(path_tok, 'import group must contain at least one name')
    else:
        item, i = _parse_import_item(toks, i)
        items.append(item)

    return ImportDecl(path=path, tok=tok_from, items=items), i


def _strip_imports(toks: list[Token]) -> tuple[list[ImportDecl], list[Token]]:
    block_openers = {KeywordType.IF, KeywordType.WHILE, KeywordType.LET, KeywordType.PROC, KeywordType.MACRO}
    imports: list[ImportDecl] = []
    body_toks: list[Token] = []
    depth = 0
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok.type == TokenType.KEYWORD:
            kw_type = cast(KeywordType, tok.value)
            if kw_type == KeywordType.FROM:
                if depth != 0:
                    error(tok, 'imports are only allowed at top level')
                import_decl, i = _parse_import_decl(toks, i)
                imports.append(import_decl)
                continue
            if kw_type == KeywordType.IMPORT:
                if depth != 0:
                    error(tok, 'imports are only allowed at top level')
                error(tok, 'only from path import name is supported')
            if kw_type in block_openers:
                depth += 1
            elif kw_type == KeywordType.END:
                depth -= 1
                if depth < 0:
                    depth = 0

        body_toks.append(tok)
        i += 1

    return imports, body_toks


def _collect_macro_body(toks: list[Token], macro_tok: Token, start_i: int) -> tuple[list[Token], int]:
    block_openers = {KeywordType.IF, KeywordType.WHILE, KeywordType.LET, KeywordType.PROC, KeywordType.MACRO}
    body: list[Token] = []
    depth = 0
    i = start_i
    while i < len(toks):
        tok = toks[i]
        if tok.type == TokenType.KEYWORD:
            kw_type = cast(KeywordType, tok.value)
            if kw_type == KeywordType.END:
                if depth == 0:
                    return body, i + 1
                depth -= 1
            elif kw_type in block_openers:
                depth += 1

        body.append(tok)
        i += 1

    error(macro_tok, f'expected a {KeywordType.END} after macro body')


def _strip_macros(toks: list[Token]) -> tuple[dict[str, Macro], list[Token]]:
    macros: dict[str, Macro] = {}
    toks_without_macros: list[Token] = []
    i = 0
    while i < len(toks):
        tok = toks[i]
        i += 1
        if tok.type == TokenType.KEYWORD and tok.value == KeywordType.MACRO:
            if i >= len(toks):
                error(tok, f'expected a name after MACRO')
            name_tok = toks[i]
            if name_tok.type != TokenType.WORD:
                error(name_tok, f'expected a name after MACRO, got {name_tok.type}')
            name = cast(str, name_tok.value)
            if name in macros:
                error(name_tok, f'macro {name} is already defined', previous_definition=macros[name].tok)
            i += 1

            body, i = _collect_macro_body(toks, tok, i)
            macros[name] = Macro(name=name, tok=tok, body=body)
        else:
            toks_without_macros.append(tok)

    return macros, toks_without_macros


def _collect_proc_toks(toks: list[Token]) -> dict[str, Token]:
    proc_toks: dict[str, Token] = {}
    depth = 0
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok.type == TokenType.KEYWORD:
            kw_type = cast(KeywordType, tok.value)
            if kw_type == KeywordType.PROC and depth == 0:
                if i + 1 >= len(toks):
                    error(tok, f'expected a name after PROC')
                name_tok = toks[i + 1]
                if name_tok.type != TokenType.WORD:
                    error(name_tok, f'expected a name after PROC, got {name_tok.type}')
                name = cast(str, name_tok.value)
                if name not in proc_toks:
                    proc_toks[name] = name_tok
                depth += 1
            elif kw_type in {KeywordType.IF, KeywordType.WHILE, KeywordType.LET}:
                depth += 1
            elif kw_type == KeywordType.END:
                depth -= 1
                if depth < 0:
                    depth = 0
        i += 1
    return proc_toks


def _top_level_proc_tokens(toks: list[Token]) -> list[Token]:
    proc_toks: list[Token] = []
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok.type != TokenType.KEYWORD or tok.value != KeywordType.PROC:
            i += 1
            continue

        depth = 0
        while i < len(toks):
            proc_tok = toks[i]
            proc_toks.append(proc_tok)
            if proc_tok.type == TokenType.KEYWORD:
                kw_type = cast(KeywordType, proc_tok.value)
                if kw_type in {KeywordType.PROC, KeywordType.IF, KeywordType.WHILE, KeywordType.LET}:
                    depth += 1
                elif kw_type == KeywordType.END:
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
            i += 1

    return proc_toks


def _validate_macro_body(macro: Macro) -> None:
    @dataclass
    class Block:
        type: KeywordType
        has_do: bool = False
        has_else: bool = False

    blocks: list[Block] = []
    for tok in macro.body:
        if tok.type != TokenType.KEYWORD:
            continue
        kw_type = cast(KeywordType, tok.value)
        match kw_type:
            case KeywordType.PROC:
                error(tok, f'PROC should not be inside of a MACRO')
            case KeywordType.MACRO:
                error(tok, f'MACRO should not be inside of another MACRO')
            case KeywordType.FROM | KeywordType.IMPORT:
                error(tok, 'imports are only allowed at top level')
            case KeywordType.IF | KeywordType.WHILE | KeywordType.LET:
                blocks.append(Block(kw_type))
            case KeywordType.DO:
                if not blocks:
                    error(tok, f'DO should follow an IF or WHILE')
                block = blocks[-1]
                if block.has_do:
                    error(tok, f'DO should follow an IF or WHILE')
                block.has_do = True
            case KeywordType.ELSE:
                if not blocks or blocks[-1].type != KeywordType.IF:
                    error(tok, f'ELSE should follow an IF')
                block = blocks[-1]
                if not block.has_do or block.has_else:
                    error(tok, f'ELSE should follow an IF')
                block.has_else = True
            case KeywordType.END:
                if not blocks:
                    error(tok, f'END should follow an IF or WHILE')
                block = blocks.pop()
                if block.type in {KeywordType.IF, KeywordType.WHILE, KeywordType.LET} and not block.has_do:
                    error(
                        tok, f'{block.type.value if isinstance(block.type.value, str) else block.type} block needs DO'
                    )
            case KeywordType.TYPE_DELIM | KeywordType.AS:
                pass

    if blocks:
        error(macro.tok, f'unclosed blocks in macro {macro.name}', blocks=blocks)


def _prepare_module_source(
    *,
    module_id: str,
    path: str,
    canonical_path: Path | None,
    toks: list[Token],
) -> ModuleSource:
    imports, toks_without_imports = _strip_imports(toks)
    macros, body_toks = _strip_macros(toks_without_imports)
    for macro in macros.values():
        _validate_macro_body(macro)
    return ModuleSource(
        id=module_id,
        path=path,
        canonical_path=canonical_path,
        toks=toks,
        imports=imports,
        body_toks=body_toks,
        macros=macros,
        proc_toks=_collect_proc_toks(body_toks),
    )


def _root_dir_from_tokens(toks: list[Token]) -> Path:
    for tok in toks:
        path = Path(tok.loc.file)
        if tok.loc.file.startswith('<'):
            break
        return path.parent.resolve(strict=False)
    return Path.cwd().resolve(strict=False)


def _module_id(root_dir: Path, path: Path) -> str:
    path = path.resolve(strict=False)
    try:
        return str(path.relative_to(root_dir))
    except ValueError:
        return str(path)


def _load_import_modules(
    root_dir: Path,
    source: ModuleSource,
    modules_by_path: dict[Path, ModuleSource],
    loading_stack: list[Path],
) -> None:
    for import_decl in source.imports:
        import_path = _normalize_import_path(root_dir, import_decl.path)
        if import_path in loading_stack:
            chain = [_display_path(path) for path in loading_stack]
            chain.append(_display_path(import_path))
            error(import_decl.tok, 'cyclic import', import_chain=chain)
        if import_path in modules_by_path:
            continue
        if not import_path.exists():
            error(import_decl.tok, f'imported file does not exist: {import_decl.path}')

        text = import_path.read_text()
        display_path = _display_path(import_path)
        toks = tokenize(text, filename=display_path)
        imported_source = _prepare_module_source(
            module_id=_module_id(root_dir, import_path),
            path=display_path,
            canonical_path=import_path,
            toks=toks,
        )
        modules_by_path[import_path] = imported_source
        _load_import_modules(root_dir, imported_source, modules_by_path, [*loading_stack, import_path])


def _resolve_module_scopes(root_dir: Path, sources: list[ModuleSource]) -> None:
    source_by_path = {source.canonical_path: source for source in sources if source.canonical_path is not None}
    resolved: set[str] = set()

    def add_imported_symbol(source: ModuleSource, alias: str, symbol: Symbol, tok: Token) -> None:
        if alias in source.proc_toks or alias in source.macros:
            error(tok, f'name {alias} is already defined')
        existing = source.imported_symbols.get(alias)
        if existing is not None:
            if existing == symbol:
                return
            error(tok, f'name {alias} is already defined')
        source.imported_symbols[alias] = symbol

    def resolve(source: ModuleSource) -> None:
        if source.id in resolved:
            return
        for import_decl in source.imports:
            imported_path = _normalize_import_path(root_dir, import_decl.path)
            imported_source = source_by_path.get(imported_path)
            if imported_source is None:
                error(import_decl.tok, f'imported file does not exist: {import_decl.path}')
            resolve(imported_source)
            for item in import_decl.items:
                symbol = imported_source.scope.get(item.name)
                if symbol is None:
                    error(item.tok, f'module {import_decl.path} does not export {item.name}')
                add_imported_symbol(source, item.alias, symbol, item.tok)

        scope = dict(source.imported_symbols)
        for name in source.proc_toks:
            scope[name] = Symbol(SymbolType.PROC, source.id, name)
        for name in source.macros:
            scope[name] = Symbol(SymbolType.MACRO, source.id, name)
        source.scope = scope
        resolved.add(source.id)

    for source in sources:
        resolve(source)


def _compile_module_source(
    source: ModuleSource,
    sources_by_id: dict[str, ModuleSource],
    *,
    is_root: bool,
    trace_ir: bool,
) -> Module:
    module = Module(id=source.id, path=source.path)

    implicit_main = False
    cur_proc: Proc | None = None

    def add_instr(instr: Instruction) -> int:
        nonlocal cur_proc
        nonlocal implicit_main
        if cur_proc is None:
            cur_proc = module.get_proc_by_name('main')
            implicit_main = True
        if cur_proc is None:
            cur_proc = Proc(name='main', contract=Contract(ins=[], outs=[]), instrs=[], tok=instr.tok)
            module.procs.append(cur_proc)
        cur_proc.instrs.append(instr)
        return len(cur_proc.instrs) - 1

    if len('abc') == 4:
        cur_proc = Proc(
            name='lol', contract=Contract(ins=[], outs=[]), instrs=[], tok=Token(TokenType.IMAGINARY, ..., loc_unknown)
        )

    @dataclass
    class Block:
        type: InstructionType
        ip1: int = -1
        ip2: int = -1
        ip3: int = -1
        ip4: int = -1
        bound_vars: list[str] = field(default_factory=list)

    block_stack: list[Block] = []

    def parse_type(tok: Token) -> ValueCls:
        if tok.type != TokenType.WORD:
            error(tok, f'expected a type after proc name, got {tok.type}')
        match tok.value:
            case 'int':
                return ValueCls(ValueClsType.INT)
            case 'bool':
                return ValueCls(ValueClsType.BOOL)
            case 'ptr':
                return ValueCls(ValueClsType.PTR)
            case _:
                error(tok, f'expected a type after proc name, got {tok.value}')

    def compile_stream(toks: list[Token], scope_source: ModuleSource, expansion_stack: list[Symbol]) -> None:
        nonlocal cur_proc
        i = 0
        while i < len(toks):
            tok = toks[i]
            i += 1
            match tok.type:
                case TokenType.IMAGINARY:
                    warn(tok, f'encountered imaginary token: {pp(tok)}')

                case TokenType.INT:
                    _ = add_instr(Instruction(type=InstructionType.PUSH_INT, tok=tok, arg1=tok.value))

                case TokenType.BOOL:
                    _ = add_instr(Instruction(type=InstructionType.PUSH_BOOL, tok=tok, arg1=tok.value))

                case TokenType.CHAR:
                    if len(tok.value) != 1:
                        error(
                            tok,
                            f'invalid character literal: {tok.value}, it must contain exactly one character',
                        )
                    _ = add_instr(Instruction(type=InstructionType.PUSH_INT, tok=tok, arg1=ord(tok.value)))

                case TokenType.STR:
                    notimplemented(tok.loc, f'string literals')

                case TokenType.KEYWORD:
                    kw_type = cast(KeywordType, tok.value)
                    match kw_type:
                        case KeywordType.IF:
                            b = Block(InstructionType.IF)
                            block_stack.append(b)
                            b.ip1 = add_instr(Instruction(type=InstructionType.IF, tok=tok))

                        case KeywordType.ELSE:
                            if not block_stack:
                                error(tok, f'ELSE should follow an IF')
                            b = block_stack[-1]
                            if b.type != InstructionType.IF:
                                error(tok, f'ELSE should follow an IF, not {pp(b.type)}')
                            b.ip3 = add_instr(Instruction(type=InstructionType.ELSE, tok=tok, arg1=missing))

                        case KeywordType.WHILE:
                            b = Block(InstructionType.WHILE)
                            block_stack.append(b)
                            b.ip1 = add_instr(Instruction(type=InstructionType.WHILE, tok=tok))

                        case KeywordType.DO:
                            if not block_stack:
                                error(tok, f'DO should follow an IF or WHILE')
                            b = block_stack[-1]
                            if b.type == InstructionType.WHILE:
                                b.ip2 = add_instr(Instruction(type=InstructionType.DO, tok=tok, arg1=missing))

                            elif b.type == InstructionType.IF:
                                b.ip2 = add_instr(Instruction(type=InstructionType.DO, tok=tok, arg1=missing))

                            else:
                                error(tok, f'DO should follow an IF or WHILE, not {pp(b.type)}')

                        case KeywordType.LET:
                            words: list[str] = []
                            while i < len(toks) and toks[i].type == TokenType.WORD:
                                words.append(toks[i].value)
                                i += 1

                            if i >= len(toks):
                                error(tok, f'expected a {KeywordType.DO} after let binding')

                            if toks[i].type != TokenType.KEYWORD or toks[i].value != KeywordType.DO:
                                error(tok, f'expected a {KeywordType.DO} after let binding, got {toks[i]}')
                            i += 1

                            if len(words) < 1:
                                error(tok, f'let must have at least one word')

                            block = Block(InstructionType.BIND, bound_vars=words)
                            block_stack.append(block)

                            for word in reversed(words):
                                _ = add_instr(Instruction(type=InstructionType.BIND, tok=tok, arg1=word))

                        case KeywordType.END:
                            if not block_stack:
                                error(tok, f'END should follow an IF or WHILE')
                            b = block_stack.pop()

                            match b.type:
                                case InstructionType.IF:
                                    if b.ip1 == -1:
                                        unreachable(tok, f'somehow the IF doesnt have an IF-instruction address saved')
                                    if b.ip2 == -1:
                                        error(tok, f'if <cond> do <body> [else <body>] end')

                                    if b.ip3 == -1:
                                        b.ip3 = add_instr(Instruction(type=InstructionType.ELSE, tok=tok, arg1=missing))

                                    assert cur_proc is not None
                                    b.ip4 = add_instr(Instruction(type=InstructionType.END, tok=tok, arg1=-1))
                                    cur_proc.instrs[-1].arg1 = b.ip4 + 1

                                    cur_proc.instrs[b.ip2].arg1 = b.ip3
                                    cur_proc.instrs[b.ip3].arg1 = b.ip4

                                case InstructionType.WHILE:
                                    if b.ip1 == -1:
                                        unreachable(
                                            tok,
                                            f'somehow the WHILE doesnt have an WHILE-instruction address saved',
                                        )
                                    if b.ip2 == -1:
                                        error(tok, f'while <cond> do <body> end')

                                    assert cur_proc is not None
                                    cur_proc.instrs[b.ip2].arg1 = b.ip3 = add_instr(
                                        Instruction(type=InstructionType.END, tok=tok, arg1=b.ip1)
                                    )

                                case InstructionType.PROC:
                                    _ = add_instr(Instruction(type=InstructionType.RET, tok=tok))
                                    cur_proc = None

                                case InstructionType.BIND:
                                    for word in b.bound_vars:
                                        _ = add_instr(Instruction(type=InstructionType.UNBIND, tok=tok, arg1=word))

                                case _:
                                    error(tok, f'END should follow an IF or WHILE, not {pp(b.type)}')

                        case KeywordType.TYPE_DELIM:
                            error(tok, f'{kw_type} should not be used only in proc signatures')

                        case KeywordType.PROC:
                            if scope_source.id != source.id:
                                error(tok, f'PROC should not be inside of a MACRO')

                            if cur_proc is not None:
                                error(tok, f'PROC should not be inside of another PROC: {cur_proc.name}')

                            if i >= len(toks):
                                error(tok, f'expected a name after PROC')
                            tok = toks[i]
                            if tok.type != TokenType.WORD:
                                error(tok, f'expected a name after PROC, got {tok.type}')
                            name = tok.value
                            i += 1

                            ins: list[ValueCls] = []
                            while i < len(toks) and toks[i].type == TokenType.WORD:
                                ins.append(parse_type(toks[i]))
                                i += 1

                            if i >= len(toks):
                                error(tok, f'expected a {KeywordType.TYPE_DELIM} after proc args')

                            if toks[i].type != TokenType.KEYWORD or toks[i].value != KeywordType.TYPE_DELIM:
                                error(tok, f'expected a {KeywordType.TYPE_DELIM} after proc args, got {toks[i]}')
                            i += 1

                            outs: list[ValueCls] = []
                            while i < len(toks) and toks[i].type == TokenType.WORD:
                                outs.append(parse_type(toks[i]))
                                i += 1

                            if i >= len(toks):
                                error(tok, f'expected a DO after proc args')
                            tok = toks[i]
                            if tok.type != TokenType.KEYWORD or tok.value != KeywordType.DO:
                                error(tok, f'expected a DO after proc args, got {tok}')
                            i += 1

                            if name == 'main' and (len(ins) != 0 or len(outs) != 0):
                                error(tok, f'main must take no args and return no values')

                            cur_proc = Proc(name=name, contract=Contract(ins=ins, outs=outs), instrs=[], tok=tok)
                            module.procs.append(cur_proc)
                            block = Block(InstructionType.PROC)
                            block_stack.append(block)

                        case KeywordType.MACRO:
                            error(tok, f'MACRO should not be inside of another MACRO')

                        case KeywordType.FROM | KeywordType.IMPORT:
                            error(tok, 'imports are only allowed at top level')

                        case KeywordType.AS:
                            error(tok, f'AS should only be used in import declarations')

                        case _:
                            assert_never(kw_type)

                case TokenType.WORD:
                    symbol = scope_source.scope.get(tok.value)
                    if symbol is not None and symbol.type == SymbolType.MACRO:
                        if symbol in expansion_stack:
                            error(
                                tok,
                                f'recursive macro expansion for {symbol.name}',
                                expansion_stack=[*expansion_stack, symbol],
                            )
                        macro_source = sources_by_id[symbol.module]
                        macro = macro_source.macros[symbol.name]
                        compile_stream(macro.body, macro_source, [*expansion_stack, symbol])
                        continue

                    expect_enum_size(IntrinsicType, 48)
                    match tok.value:
                        case _ if tok.value in INTRINSIC_TO_INTRINSIC_TYPE:
                            _ = add_instr(
                                Instruction(
                                    type=InstructionType.INTRINSIC,
                                    tok=tok,
                                    arg1=INTRINSIC_TO_INTRINSIC_TYPE[tok.value],
                                )
                            )

                        case 'int':
                            _ = add_instr(
                                Instruction(type=InstructionType.PUSH_TYPE, tok=tok, arg1=ValueCls(ValueClsType.INT))
                            )
                        case 'bool':
                            _ = add_instr(
                                Instruction(type=InstructionType.PUSH_TYPE, tok=tok, arg1=ValueCls(ValueClsType.BOOL))
                            )
                        case 'ptr':
                            _ = add_instr(
                                Instruction(type=InstructionType.PUSH_TYPE, tok=tok, arg1=ValueCls(ValueClsType.PTR))
                            )

                        case _:
                            for block in reversed(block_stack):
                                if block.type != InstructionType.BIND:
                                    continue
                                if tok.value in block.bound_vars:
                                    _ = add_instr(
                                        Instruction(
                                            type=InstructionType.LOAD_BIND,
                                            tok=tok,
                                            arg1=tok.value,
                                        )
                                    )
                                    break
                            else:
                                if symbol is not None and symbol.type == SymbolType.PROC:
                                    ref = ProcRef(symbol.module, symbol.name)
                                else:
                                    ref = ProcRef(scope_source.id, tok.value)
                                _ = add_instr(Instruction(type=InstructionType.WORD, tok=tok, arg1=ref))

                case _:
                    assert_never(tok.type)

    toks_to_compile = source.body_toks if is_root else _top_level_proc_tokens(source.body_toks)
    compile_stream(toks_to_compile, source, [])

    if implicit_main:
        _ = add_instr(Instruction(type=InstructionType.RET, tok=Token(TokenType.IMAGINARY, '', loc_unknown)))

    if block_stack:
        error(None, 'unclosed blocks', blocks=block_stack)

    if is_root and not module.get_proc_by_name('main'):
        module.procs.append(
            Proc(
                name='main',
                contract=Contract(ins=[], outs=[]),
                instrs=[],
                tok=Token(TokenType.KEYWORD, value=KeywordType.PROC, loc=loc_unknown),
            )
        )

    if trace_ir:
        trace(loc_unknown, f'Compiled module {source.path}', module=module)
    return module


def _module_compile_order(
    root_source: ModuleSource, sources_by_path: dict[Path, ModuleSource], root_dir: Path
) -> list[ModuleSource]:
    ordered: list[ModuleSource] = []
    seen: set[str] = set()

    def visit(source: ModuleSource) -> None:
        if source.id in seen:
            return
        seen.add(source.id)
        for import_decl in source.imports:
            imported_path = _normalize_import_path(root_dir, import_decl.path)
            imported_source = sources_by_path.get(imported_path)
            if imported_source is not None:
                visit(imported_source)
        ordered.append(source)

    visit(root_source)
    return ordered


def compile(toks: list[Token]) -> IR:
    root_dir = _root_dir_from_tokens(toks)
    root_source = _prepare_module_source(
        module_id='',
        path='<root>',
        canonical_path=None,
        toks=toks,
    )
    modules_by_path: dict[Path, ModuleSource] = {}
    _load_import_modules(root_dir, root_source, modules_by_path, [])
    sources = [root_source, *modules_by_path.values()]
    _resolve_module_scopes(root_dir, sources)

    compile_order = _module_compile_order(root_source, modules_by_path, root_dir)
    sources_by_id = {source.id: source for source in sources}
    ir = IR(root_module='')
    for source in compile_order:
        ir.modules.append(
            _compile_module_source(
                source,
                sources_by_id,
                is_root=source.id == '',
                trace_ir=False,
            )
        )

    trace(
        loc_unknown,
        f'Compiled IR',
        IR=ir,
    )
    return ir


def typecheck(ir: IR) -> None:
    @dataclass
    class Block:
        type: InstructionType
        stack1: Stack | None = None
        stack2: Stack | None = None
        stack3: Stack | None = None
        stack4: Stack | None = None

    def compare_stacks(stack1: Stack, stack2: Stack) -> bool:
        if len(stack1) != len(stack2):
            return False
        for _, (e1, e2) in enumerate(zip(stack1, stack2)):
            if e1.type != e2.type:
                return False
        return True

    def instr_contract_name(instr: Instruction) -> object:
        if instr.type == InstructionType.WORD:
            ref = cast(ProcRef, instr.arg1)
            return ref.name
        return instr.arg1

    def typecheck_contract(instr: Instruction, stack: Stack, contract: Contract) -> None:
        contract_name = instr_contract_name(instr)
        # 1. check that stack and ins are the same
        if len(stack) < len(contract.ins):
            error(
                instr,
                f'stack too small for {contract_name}: expected {len(contract.ins)} but got {len(stack)}',
                stack=stack,
                ins=contract.ins,
            )

        stack_suffix = stack[len(stack) - len(contract.ins) :]
        for i, (e1, e2) in enumerate(zip(stack_suffix, contract.ins)):
            if e1.type != e2:
                error(
                    instr,
                    f'stack doesnt match at {i} for {contract_name}: expected {pp(e2)} but got {pp(e1.type)}',
                    stack=stack,
                    ins=contract.ins,
                )

        # 2. remove ins from stack
        for _ in contract.ins:
            _ = stack.pop()

        # 3. put outs on stack
        for out in contract.outs:
            stack.append(StackEntry(out, typechecking, tok=instr.tok))

    for proc in ir.procs:
        block_stack: list[Block] = []
        stack: Stack = []
        bound_vars: list[tuple[str, StackEntry]] = []

        for type_in in proc.contract.ins:
            stack.append(StackEntry(type_in, typechecking, tok=proc.tok))
        instrs = proc.instrs
        for instr in instrs:
            match instr.type:
                case InstructionType.PUSH_INT:
                    stack.append(StackEntry(ValueCls(ValueClsType.INT, unused), typechecking, tok=instr.tok))

                case InstructionType.PUSH_BOOL:
                    stack.append(StackEntry(ValueCls(ValueClsType.BOOL, unused), typechecking, tok=instr.tok))

                case InstructionType.PUSH_TYPE:
                    stack.append(StackEntry(ValueCls(ValueClsType.TYPE, instr.arg1), typechecking, tok=instr.tok))

                case InstructionType.WORD:
                    proc_ref = cast(ProcRef, instr.arg1)
                    proc_called = ir.get_proc_by_ref(proc_ref)
                    if proc_called is None:
                        error(instr, f'unknown word {proc_ref.name}')

                    typecheck_contract(instr, stack, proc_called.contract)

                case InstructionType.LOAD_BIND:
                    for name, stack_entry in reversed(bound_vars):
                        if name == instr.arg1:
                            stack.append(stack_entry)
                            break
                    else:
                        error(instr, f'unknown variable {instr.arg1}')

                # if // s = stack1
                #   <cond>
                # then // s bool
                #   <body>
                # else // s XS = stack2
                #   <body>
                # end // s XS
                case InstructionType.IF:
                    stack_copy = stack.copy()
                    block = Block(InstructionType.IF)
                    block.stack1 = stack_copy
                    block_stack.append(block)

                case InstructionType.ELSE:
                    block = block_stack[-1]
                    if block.type == InstructionType.IF:
                        block.stack2 = stack.copy()
                        if block.stack1 is None:
                            unreachable(instr, f'{InstructionType.IF} block doesnt have stack1 saved up')
                        stack = block.stack1

                    else:
                        error(instr, f'{InstructionType.ELSE} must come after {InstructionType.IF}')

                case InstructionType.BIND:
                    if len(stack) < 1:
                        error(instr, f'{InstructionType.BIND} must have at least 1 element on the stack')
                    bound_vars.append((instr.arg1, stack.pop()))

                case InstructionType.UNBIND:
                    if len(bound_vars) < 1:
                        unreachable(instr, f'{InstructionType.UNBIND} must come after {InstructionType.BIND}')

                    bound_var = bound_vars.pop()
                    if bound_var[0] != instr.arg1:
                        error(
                            instr, f'{InstructionType.UNBIND} must unbind the same variable as {InstructionType.BIND}'
                        )

                case InstructionType.END:
                    block = block_stack.pop()
                    if block.type == InstructionType.IF:
                        if block.stack2 is None:
                            unreachable(instr, f'{InstructionType.IF} block doesnt have stack2 saved up')
                        if not compare_stacks(block.stack2, stack):
                            error(
                                instr,
                                f'both branches of {InstructionType.IF} must leave the stack in the same state',
                                stack_then=block.stack2,
                                stack_else=stack,
                            )

                    elif block.type == InstructionType.WHILE:
                        if block.stack1 is None:
                            unreachable(instr, f'{InstructionType.WHILE} block doesnt have stack1 saved up')
                        if not compare_stacks(block.stack1, stack):
                            error(instr, f'{InstructionType.WHILE} must not alter the stack state')

                    else:
                        error(
                            instr,
                            f'{InstructionType.END} must come after {InstructionType.IF} or {InstructionType.WHILE}',
                        )

                # while // s = stack1
                #   <cond>
                # do // s bool
                #   <body>
                # end // s
                case InstructionType.WHILE:
                    stack_copy = stack.copy()
                    block = Block(InstructionType.WHILE)
                    block.stack1 = stack_copy
                    block_stack.append(block)

                case InstructionType.DO:
                    if len(stack) < 1:
                        error(
                            instr,
                            f'not enough items on the stack for {InstructionType.DO}: it expects one BOOL on the stack',
                        )
                    a = stack.pop()
                    if a.type.type == ValueClsType.BOOL:
                        block = block_stack[-1]
                        if block.type == InstructionType.IF:
                            if block.stack1 is None:
                                unreachable(instr, f'{InstructionType.IF} block doesnt have stack1 saved up')
                            if not compare_stacks(block.stack1, stack):
                                error(
                                    instr,
                                    f'condition part of {InstructionType.IF} should put exactly one BOOL on the stack',
                                    expected_stack=block.stack1,
                                    actual_stack=stack,
                                )

                        elif block.type == InstructionType.WHILE:
                            if block.stack1 is None:
                                unreachable(instr, f'{InstructionType.WHILE} block doesnt have stack1 saved up')
                            if not compare_stacks(block.stack1, stack):
                                error(
                                    instr,
                                    f'condition part of {InstructionType.WHILE} should put exactly one BOOL on the stack',
                                    expected_stack=block.stack1,
                                    actual_stack=stack,
                                )

                        else:
                            error(
                                instr,
                                f'{InstructionType.DO} must come after {InstructionType.IF} or {InstructionType.WHILE}',
                            )

                    else:
                        error(
                            instr,
                            f'{InstructionType.DO} expects one BOOL on the stack, but got {pp(a.type)}',
                            other_stack_items=stack,
                        )

                case InstructionType.PROC:
                    notimplemented(instr, f'typechecking {InstructionType.PROC}')

                case InstructionType.RET:
                    if len(stack) != len(proc.contract.outs):
                        error(
                            instr,
                            f'return type mismatch: expected {len(proc.contract.outs)} items on the stack, got {len(stack)}',
                            stack=stack,
                            outs=proc.contract.outs,
                        )

                    for i, (item_stack, item_out) in enumerate(zip(stack, proc.contract.outs)):
                        t1 = item_stack.type
                        t2 = item_out
                        if t1 != t2:
                            error(
                                instr,
                                f'return type mismatch at {i}: expected {t2} on the stack, got {t1}',
                                stack=stack,
                                outs=proc.contract.outs,
                            )

                    for _ in proc.contract.outs:
                        _ = stack.pop()

                case InstructionType.LABEL:
                    notimplemented(instr, f'typechecking {InstructionType.LABEL}')

                case InstructionType.INTRINSIC:
                    intr_type = cast(IntrinsicType, instr.arg1)
                    match intr_type:
                        # arithmetic:
                        case IntrinsicType.ADD | IntrinsicType.SUB:
                            if len(stack) < 2:
                                typecheck_contract(
                                    instr,
                                    stack=stack,
                                    contract=Contract(
                                        ins=[ValueCls(ValueClsType.INT), ValueCls(ValueClsType.INT)],
                                        outs=[ValueCls(ValueClsType.INT)],
                                    ),
                                )

                            b = stack[-1]
                            a = stack[-2]
                            if a.type.type == ValueClsType.PTR and b.type.type == ValueClsType.INT:
                                _ = stack.pop()
                                _ = stack.pop()
                                stack.append(StackEntry(ValueCls(ValueClsType.PTR), typechecking, tok=instr.tok))
                            else:
                                typecheck_contract(
                                    instr,
                                    stack,
                                    Contract(
                                        ins=[ValueCls(ValueClsType.INT), ValueCls(ValueClsType.INT)],
                                        outs=[ValueCls(ValueClsType.INT)],
                                    ),
                                )

                        case IntrinsicType.MUL | IntrinsicType.DIV:
                            typecheck_contract(
                                instr,
                                stack,
                                Contract(
                                    ins=[ValueCls(ValueClsType.INT), ValueCls(ValueClsType.INT)],
                                    outs=[ValueCls(ValueClsType.INT)],
                                ),
                            )

                        case IntrinsicType.MOD:
                            typecheck_contract(
                                instr,
                                stack,
                                Contract(
                                    ins=[ValueCls(ValueClsType.INT), ValueCls(ValueClsType.INT)],
                                    outs=[ValueCls(ValueClsType.INT)],
                                ),
                            )

                        case IntrinsicType.DIVMOD:
                            typecheck_contract(
                                instr,
                                stack,
                                Contract(
                                    ins=[ValueCls(ValueClsType.INT), ValueCls(ValueClsType.INT)],
                                    outs=[ValueCls(ValueClsType.INT), ValueCls(ValueClsType.INT)],
                                ),
                            )

                        case IntrinsicType.SHL | IntrinsicType.SHR:
                            typecheck_contract(
                                instr,
                                stack,
                                Contract(
                                    ins=[ValueCls(ValueClsType.INT), ValueCls(ValueClsType.INT)],
                                    outs=[ValueCls(ValueClsType.INT)],
                                ),
                            )

                        case IntrinsicType.BOR | IntrinsicType.BAND | IntrinsicType.BXOR:
                            typecheck_contract(
                                instr,
                                stack,
                                Contract(
                                    ins=[ValueCls(ValueClsType.INT), ValueCls(ValueClsType.INT)],
                                    outs=[ValueCls(ValueClsType.INT)],
                                ),
                            )

                        case IntrinsicType.BNOT:
                            typecheck_contract(
                                instr,
                                stack,
                                Contract(ins=[ValueCls(ValueClsType.INT)], outs=[ValueCls(ValueClsType.INT)]),
                            )

                        # logic:
                        case IntrinsicType.AND | IntrinsicType.OR:
                            typecheck_contract(
                                instr,
                                stack,
                                Contract(
                                    ins=[ValueCls(ValueClsType.BOOL), ValueCls(ValueClsType.BOOL)],
                                    outs=[ValueCls(ValueClsType.BOOL)],
                                ),
                            )

                        case IntrinsicType.NOT:
                            typecheck_contract(
                                instr,
                                stack,
                                Contract(ins=[ValueCls(ValueClsType.BOOL)], outs=[ValueCls(ValueClsType.BOOL)]),
                            )

                        # comparison:
                        case IntrinsicType.EQ | IntrinsicType.NE:
                            typecheck_contract(
                                instr,
                                stack,
                                Contract(
                                    ins=[ValueCls(ValueClsType.INT), ValueCls(ValueClsType.INT)],
                                    outs=[ValueCls(ValueClsType.BOOL)],
                                ),
                            )

                        case IntrinsicType.LT | IntrinsicType.LE | IntrinsicType.GT | IntrinsicType.GE:
                            typecheck_contract(
                                instr,
                                stack,
                                Contract(
                                    ins=[ValueCls(ValueClsType.INT), ValueCls(ValueClsType.INT)],
                                    outs=[ValueCls(ValueClsType.BOOL)],
                                ),
                            )

                        # stack manipulation:
                        case IntrinsicType.DUP:
                            # a -- a a
                            if len(stack) < 1:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects one item on the stack',
                                    stack=stack,
                                )
                            a = stack.pop()
                            stack.append(a)
                            stack.append(a)

                        case IntrinsicType.DUP2:
                            # a b -- a b a b
                            if len(stack) < 2:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects two items on the stack',
                                    stack=stack,
                                )
                            b = stack.pop()
                            a = stack.pop()
                            stack.append(a)
                            stack.append(b)
                            stack.append(a)
                            stack.append(b)

                        case IntrinsicType.DROP:
                            # a --
                            if len(stack) < 1:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects one item on the stack',
                                    stack=stack,
                                )
                            _ = stack.pop()

                        case IntrinsicType.SWAP:
                            # a b -- b a
                            if len(stack) < 2:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects two items on the stack',
                                    stack=stack,
                                )
                            b = stack.pop()
                            a = stack.pop()
                            stack.append(b)
                            stack.append(a)

                        case IntrinsicType.SWAP2:
                            # a b x y -- x y a b
                            if len(stack) < 4:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects four items on the stack',
                                    stack=stack,
                                )
                            y = stack.pop()
                            x = stack.pop()
                            b = stack.pop()
                            a = stack.pop()
                            stack.append(x)
                            stack.append(y)
                            stack.append(a)
                            stack.append(b)

                        case IntrinsicType.ROT:
                            # a b c -- b c a
                            if len(stack) < 3:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects three items on the stack',
                                    stack=stack,
                                )
                            c = stack.pop()
                            b = stack.pop()
                            a = stack.pop()
                            stack.append(b)
                            stack.append(c)
                            stack.append(a)

                        case IntrinsicType.CAST:
                            # x T -- (T)x
                            if len(stack) < 2:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects two items on the stack',
                                    stack=stack,
                                )
                            t = stack.pop()
                            x = stack.pop()
                            if t.type.type != ValueClsType.TYPE:
                                error(
                                    instr,
                                    f'cast expects a type to cast to, but got {pp(t.type)}',
                                    stack=stack,
                                )

                            trace(
                                instr,
                                f'casting',
                                x=x,
                                t=t,
                                x_type=x.type,
                                x_val=x.val,
                                t_type=t.type,
                                t_val=t.val,
                                t_type_val=t.type.val,
                            )
                            match (x.type, t.type.val):
                                case _ if x.type == t.type.val:
                                    warn(
                                        instr,
                                        f'redundant cast {pp(x.type)} to {pp(t.type.val)}',
                                    )
                                    stack.append(StackEntry(t.type.val, typechecking, tok=instr.tok))

                                case (ValueCls(ValueClsType.INT), ValueCls(ValueClsType.BOOL)):
                                    stack.append(StackEntry(t.type.val, typechecking, tok=instr.tok))

                                case (ValueCls(ValueClsType.BOOL), ValueCls(ValueClsType.INT)):
                                    stack.append(StackEntry(t.type.val, typechecking, tok=instr.tok))

                                case (ValueCls(ValueClsType.INT), ValueCls(ValueClsType.PTR)):
                                    stack.append(StackEntry(t.type.val, typechecking, tok=instr.tok))

                                case (ValueCls(ValueClsType.PTR), ValueCls(ValueClsType.INT)):
                                    stack.append(StackEntry(t.type.val, typechecking, tok=instr.tok))

                                case _:
                                    error(
                                        instr,
                                        f'cannot cast {pp(x.type)} to {pp(t.type.val)}',
                                        stack=stack,
                                    )

                        # memory:
                        case IntrinsicType.ALLOC:
                            typecheck_contract(
                                instr,
                                stack,
                                Contract(ins=[ValueCls(ValueClsType.INT)], outs=[ValueCls(ValueClsType.PTR)]),
                            )

                        case (
                            IntrinsicType.READ_I8
                            | IntrinsicType.READ_I16
                            | IntrinsicType.READ_I32
                            | IntrinsicType.READ_I64
                            | IntrinsicType.READ_U8
                            | IntrinsicType.READ_U16
                            | IntrinsicType.READ_U32
                            | IntrinsicType.READ_U64
                        ):
                            typecheck_contract(
                                instr,
                                stack,
                                Contract(ins=[ValueCls(ValueClsType.PTR)], outs=[ValueCls(ValueClsType.INT)]),
                            )

                        case (
                            IntrinsicType.WRITE_I8
                            | IntrinsicType.WRITE_I16
                            | IntrinsicType.WRITE_I32
                            | IntrinsicType.WRITE_I64
                            | IntrinsicType.WRITE_U8
                            | IntrinsicType.WRITE_U16
                            | IntrinsicType.WRITE_U32
                            | IntrinsicType.WRITE_U64
                        ):
                            typecheck_contract(
                                instr,
                                stack,
                                Contract(ins=[ValueCls(ValueClsType.INT), ValueCls(ValueClsType.PTR)], outs=[]),
                            )

                        # debugging:
                        case IntrinsicType.PRINT:
                            # a --
                            if len(stack) < 1:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects one item on the stack',
                                    stack=stack,
                                )
                            a = stack.pop()

                        case IntrinsicType.PUTC:
                            typecheck_contract(
                                instr,
                                stack,
                                Contract(ins=[ValueCls(ValueClsType.INT)], outs=[]),
                            )

                        case IntrinsicType.DEBUG:
                            info(
                                instr,
                                'Stack at compile time:',
                                stack=stack,
                            )

                        case _:
                            assert_never(intr_type)
                case _:
                    assert_never(instr.type)

        if stack:
            error(
                None,
                f'stack is not empty: {len(stack)} items are unhandled',
                stack=stack,
            )

        if block_stack:
            unreachable(
                None,
                block_stack=block_stack,
            )


def interpret(ir: IR) -> None:

    @dataclass
    class Frame:
        instrs: list[Instruction]
        ip: int = 0

    main = ir.get_root_module().get_proc_by_name('main')
    assert main is not None

    stack: Stack = []
    bound_vars: list[tuple[str, StackEntry]] = []
    frame_stack: list[Frame] = [Frame(main.instrs)]

    while frame_stack and frame_stack[-1].ip < len(frame_stack[-1].instrs):
        frame = frame_stack[-1]
        instr = frame.instrs[frame.ip]
        frame.ip += 1
        match instr.type:
            case InstructionType.PUSH_INT:
                stack.append(StackEntry(ValueCls(ValueClsType.INT, unused), instr.arg1, tok=instr.tok))

            case InstructionType.PUSH_BOOL:
                stack.append(StackEntry(ValueCls(ValueClsType.BOOL, unused), instr.arg1, tok=instr.tok))

            case InstructionType.PUSH_TYPE:
                stack.append(StackEntry(ValueCls(ValueClsType.TYPE, instr.arg1), instr.arg1, tok=instr.tok))

            case InstructionType.WORD:
                proc_ref = cast(ProcRef, instr.arg1)
                proc = ir.get_proc_by_ref(proc_ref)
                if proc is None:
                    typecheck_has_a_bug(instr, f'unknown word: {proc_ref.name}')
                frame_stack.append(Frame(proc.instrs))

            case InstructionType.IF:
                pass

            case InstructionType.ELSE:
                frame.ip = instr.arg1

            case InstructionType.END:
                frame.ip = instr.arg1

            case InstructionType.WHILE:
                pass

            case InstructionType.DO:
                if len(stack) < 1:
                    typecheck_has_a_bug(instr, 'not enough items on stack')
                a = stack.pop()
                if a.type.type == ValueClsType.BOOL:
                    if a.val:
                        pass
                    else:
                        frame.ip = instr.arg1 + 1

                else:
                    typecheck_has_a_bug(instr, f'expected BOOL, got {pp(a.type)}')

            case InstructionType.PROC:
                unreachable(instr, 'PROC instructions must not be emitted')

            case InstructionType.RET:
                _ = frame_stack.pop()

            case InstructionType.LABEL:
                notimplemented(instr, 'labels are not implemented yet')

            case InstructionType.BIND:
                bound_vars.append((instr.arg1, stack.pop()))

            case InstructionType.UNBIND:
                _ = bound_vars.pop()

            case InstructionType.LOAD_BIND:
                for name, value in reversed(bound_vars):
                    if name == instr.arg1:
                        stack.append(value)
                        break

            case InstructionType.INTRINSIC:
                intr_type = cast(IntrinsicType, instr.arg1)
                match intr_type:
                    # arithmetic:
                    case IntrinsicType.ADD:
                        # a b -- a+b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(StackEntry(a.type, a.val + b.val, tok=instr.tok))

                        elif a.type.type == ValueClsType.PTR and b.type.type == ValueClsType.INT:
                            if not isinstance(a.val, MemoryPtr):
                                error(instr, f'cannot do pointer arithmetic on non-allocated pointer {a.val}')
                            stack.append(StackEntry(a.type, MemoryPtr(a.val.buf, a.val.offset + b.val), tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected INT INT or PTR INT, got {pp(a.type)} and {pp(b.type)}'
                            )

                    case IntrinsicType.SUB:
                        # a b -- a-b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(StackEntry(a.type, a.val - b.val, tok=instr.tok))

                        elif a.type.type == ValueClsType.PTR and b.type.type == ValueClsType.INT:
                            if not isinstance(a.val, MemoryPtr):
                                error(instr, f'cannot do pointer arithmetic on non-allocated pointer {a.val}')
                            stack.append(StackEntry(a.type, MemoryPtr(a.val.buf, a.val.offset - b.val), tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected INT INT or PTR INT, got {pp(a.type)} and {pp(b.type)}'
                            )

                    case IntrinsicType.MUL:
                        # a b -- a*b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(StackEntry(a.type, a.val * b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.DIV:
                        # a b -- a/b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            if b.val == 0:
                                error(instr, 'division by zero', a=a, b=b)
                            stack.append(StackEntry(a.type, a.val // b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.MOD:
                        # a b -- a%b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(StackEntry(a.type, a.val % b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.DIVMOD:
                        # a b -- a//b a%b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(StackEntry(a.type, a.val // b.val, tok=instr.tok))
                            stack.append(StackEntry(a.type, a.val % b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.SHL:
                        # a b -- a<<b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(StackEntry(a.type, a.val << b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.SHR:
                        # a b -- a>>b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(StackEntry(a.type, a.val >> b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.BOR:
                        # a b -- a|b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(StackEntry(a.type, a.val | b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.BAND:
                        # a b -- a&b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(StackEntry(a.type, a.val & b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.BXOR:
                        # a b -- a^b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(StackEntry(a.type, a.val ^ b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.BNOT:
                        # a -- ~a
                        if len(stack) < 1:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == ValueClsType.INT:
                            stack.append(StackEntry(a.type, ~a.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected an INT, got {pp(a.type)}')

                    # logic:
                    case IntrinsicType.AND:
                        # a b -- (a and b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == b.type.type == ValueClsType.BOOL:
                            stack.append(StackEntry(a.type, a.val and b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two BOOLs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.OR:
                        # a b -- (a or b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == b.type.type == ValueClsType.BOOL:
                            stack.append(StackEntry(a.type, a.val or b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two BOOLs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.NOT:
                        # a -- (not a)
                        if len(stack) < 1:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == ValueClsType.BOOL:
                            stack.append(StackEntry(a.type, not a.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected a BOOL, got {pp(a.type)}')

                    # comparison:
                    case IntrinsicType.EQ:
                        # a b -- (a == b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(StackEntry(ValueCls(ValueClsType.BOOL), a.val == b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.NE:
                        # a b -- (a != b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(StackEntry(ValueCls(ValueClsType.BOOL), a.val != b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.LT:
                        # a b -- (a < b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(StackEntry(ValueCls(ValueClsType.BOOL), a.val < b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.GT:
                        # a b -- (a > b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(StackEntry(ValueCls(ValueClsType.BOOL), a.val > b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.LE:
                        # a b -- (a <= b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(StackEntry(ValueCls(ValueClsType.BOOL), a.val <= b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.GE:
                        # a b -- (a >= b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 4)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(StackEntry(ValueCls(ValueClsType.BOOL), a.val >= b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    # stack manipulation:
                    case IntrinsicType.DUP:
                        # a -- a a
                        if len(stack) < 1:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        a = stack.pop()
                        stack.append(a)
                        stack.append(a)

                    case IntrinsicType.DUP2:
                        # a b -- a b a b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        stack.append(a)
                        stack.append(b)
                        stack.append(a)
                        stack.append(b)

                    case IntrinsicType.DROP:
                        # a --
                        if len(stack) < 1:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        _ = stack.pop()

                    case IntrinsicType.SWAP:
                        # a b -- b a
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        stack.append(b)
                        stack.append(a)
                    case IntrinsicType.SWAP2:
                        # a b x y -- x y a b
                        if len(stack) < 4:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        y = stack.pop()
                        x = stack.pop()
                        b = stack.pop()
                        a = stack.pop()
                        stack.append(x)
                        stack.append(y)
                        stack.append(a)
                        stack.append(b)

                    case IntrinsicType.ROT:
                        # a b c -- b c a
                        if len(stack) < 3:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        c = stack.pop()
                        b = stack.pop()
                        a = stack.pop()
                        stack.append(b)
                        stack.append(c)
                        stack.append(a)

                    case IntrinsicType.CAST:
                        # x T -- (T)x
                        if len(stack) < 2:
                            error(
                                instr,
                                f'not enough items on stack for {intr_type}: it expects two items on the stack',
                                stack=stack,
                            )
                        t = stack.pop()
                        x = stack.pop()
                        if t.type.type != ValueClsType.TYPE:
                            error(
                                instr,
                                f'cast expects a type to cast to, but got {pp(t.type)}',
                                stack=stack,
                            )

                        match (x.type, t.type.val):
                            case _ if x.type == t.type.val:
                                warn(
                                    instr,
                                    f'redundant cast {pp(x.type)} to {pp(t.type.val)}',
                                )
                                stack.append(StackEntry(t.type.val, x.val, tok=instr.tok))

                            case (ValueCls(ValueClsType.INT), ValueCls(ValueClsType.BOOL)):
                                stack.append(StackEntry(t.type.val, bool(x.val), tok=instr.tok))

                            case (ValueCls(ValueClsType.BOOL), ValueCls(ValueClsType.INT)):
                                stack.append(StackEntry(t.type.val, int(x.val), tok=instr.tok))

                            case (ValueCls(ValueClsType.INT), ValueCls(ValueClsType.PTR)):
                                stack.append(StackEntry(t.type.val, x.val, tok=instr.tok))

                            case (ValueCls(ValueClsType.PTR), ValueCls(ValueClsType.INT)):
                                stack.append(StackEntry(t.type.val, x.val, tok=instr.tok))

                            case _:
                                error(
                                    instr,
                                    f'cannot cast {pp(x.type)} to {pp(t.type.val)}',
                                    stack=stack,
                                )

                    # memory:
                    case IntrinsicType.ALLOC:
                        if len(stack) < 1:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        alloc_size = stack.pop()
                        if alloc_size.type.type != ValueClsType.INT:
                            typecheck_has_a_bug(instr, f'expected an INT, got {pp(alloc_size.type)}')
                        if alloc_size.val < 0:
                            error(instr, f'cannot allocate a negative buffer size: {alloc_size.val}')
                        stack.append(
                            StackEntry(ValueCls(ValueClsType.PTR), MemoryPtr(bytearray(alloc_size.val)), tok=instr.tok)
                        )

                    case (
                        IntrinsicType.READ_I8
                        | IntrinsicType.READ_I16
                        | IntrinsicType.READ_I32
                        | IntrinsicType.READ_I64
                        | IntrinsicType.READ_U8
                        | IntrinsicType.READ_U16
                        | IntrinsicType.READ_U32
                        | IntrinsicType.READ_U64
                    ):
                        if len(stack) < 1:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        ptr = stack.pop()
                        if ptr.type.type != ValueClsType.PTR:
                            typecheck_has_a_bug(instr, f'expected a PTR, got {pp(ptr.type)}')
                        if not isinstance(ptr.val, MemoryPtr):
                            error(instr, f'cannot read from non-allocated pointer {ptr.val}')
                        bits, signed = MEMORY_ACCESS_BITS[intr_type]
                        access_size = bits // 8
                        if ptr.val.offset < 0 or ptr.val.offset + access_size > len(ptr.val.buf):
                            error(instr, f'pointer read out of bounds: offset {ptr.val.offset}, size {access_size}')
                        read_val = int.from_bytes(
                            ptr.val.buf[ptr.val.offset : ptr.val.offset + access_size], 'little', signed=signed
                        )
                        stack.append(StackEntry(ValueCls(ValueClsType.INT), read_val, tok=instr.tok))

                    case (
                        IntrinsicType.WRITE_I8
                        | IntrinsicType.WRITE_I16
                        | IntrinsicType.WRITE_I32
                        | IntrinsicType.WRITE_I64
                        | IntrinsicType.WRITE_U8
                        | IntrinsicType.WRITE_U16
                        | IntrinsicType.WRITE_U32
                        | IntrinsicType.WRITE_U64
                    ):
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        ptr = stack.pop()
                        val = stack.pop()
                        if ptr.type.type != ValueClsType.PTR or val.type.type != ValueClsType.INT:
                            typecheck_has_a_bug(instr, f'expected INT PTR, got {pp(val.type)} and {pp(ptr.type)}')
                        if not isinstance(ptr.val, MemoryPtr):
                            error(instr, f'cannot write to non-allocated pointer {ptr.val}')
                        bits, signed = MEMORY_ACCESS_BITS[intr_type]
                        access_size = bits // 8
                        if ptr.val.offset < 0 or ptr.val.offset + access_size > len(ptr.val.buf):
                            error(instr, f'pointer write out of bounds: offset {ptr.val.offset}, size {access_size}')
                        try:
                            bs = int(val.val).to_bytes(access_size, 'little', signed=signed)
                        except OverflowError:
                            sign_name = 'signed' if signed else 'unsigned'
                            error(instr, f'{val.val} does not fit in {bits}-bit {sign_name} integer')
                        ptr.val.buf[ptr.val.offset : ptr.val.offset + access_size] = bs

                    # debugging:
                    case IntrinsicType.PRINT:
                        # a --
                        if len(stack) < 1:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        a = stack.pop()
                        print(f'[PRINT] {pp(a)}')

                    case IntrinsicType.PUTC:
                        if len(stack) < 1:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        a = stack.pop()
                        if a.type.type != ValueClsType.INT:
                            typecheck_has_a_bug(instr, f'expected an INT, got {pp(a.type)}')
                        print(chr(a.val), end='')

                    case IntrinsicType.DEBUG:
                        info(instr, 'Stack at runtime:', stack=stack)

                    case _:
                        assert_never(intr_type)
            case _:
                assert_never(instr.type)

    if stack:
        typecheck_has_a_bug(None, 'stack not empty after execution')


def translate(ir: IR) -> str:

    @dataclass
    class Block:
        type: InstructionType
        stack: Stack | None = None

    _used_names: set[str] = set()

    def get_varname(hint: str) -> str:
        i = 0
        while f'{hint}_{i}' in _used_names:
            i += 1
        _used_names.add(f'{hint}_{i}')
        return f'{hint}_{i}'

    def c_type(type: ValueCls) -> str:
        match type.type:
            case ValueClsType.INT:
                return 'int'
            case ValueClsType.BOOL:
                return 'bool'
            case ValueClsType.PTR:
                return 'void*'
            case ValueClsType.TYPE:
                return f'type[{c_type(type.val)}]'
            case _:
                assert_never(type.type)

    def declare_var(var: str, type: ValueCls) -> None:
        typ: str = c_type(type)
        sb_header.add(f'  {typ} {var};\n')

    def c_str(s: str) -> str:
        return s.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

    def c_ident(s: str) -> str:
        res = ''
        for c in s:
            if 'a' <= c <= 'z' or 'A' <= c <= 'Z' or '0' <= c <= '9':
                res += c
            else:
                res += f'_{ord(c):x}_'
        return res

    def c_proc_ident(module_id: str, proc_name: str) -> str:
        if module_id == ir.root_module:
            return c_ident(proc_name)
        return c_ident(f'{module_id}__{proc_name}')

    def emit_division_by_zero_check(instr: Instruction, a: StackEntry, b: StackEntry) -> None:
        loc = c_str(repr(instr.tok.loc))
        loc_a = c_str(repr(a.tok.loc))
        loc_b = c_str(repr(b.tok.loc))
        sb.add(f'{'':{indent}}if ({b.val} == 0) {{\n')
        sb.add(f'{'':{indent + 2}}printf("[ERROR] {loc}: division by zero\\n");\n')
        sb.add(f'{'':{indent + 2}}printf("[NOTE] a: INT():%d@{loc_a}\\n", {a.val});\n')
        sb.add(f'{'':{indent + 2}}printf("[NOTE] b: INT():%d@{loc_b}\\n", {b.val});\n')
        sb.add(f'{'':{indent + 2}}exit(1);\n')
        sb.add(f'{'':{indent}}}}\n')

    def copy_stacks(src: Stack, dst: Stack) -> None:
        for a, b in zip(src, dst):
            if a.type != b.type:
                unreachable(instr)

            if a.val == b.val:
                continue

            expect_enum_size(ValueClsType, 4)
            if a.type.type == b.type.type == ValueClsType.INT:
                sb.add(f'{'':{indent}}{b.val} = {a.val};\n')

            elif a.type.type == b.type.type == ValueClsType.BOOL:
                sb.add(f'{'':{indent}}{b.val} = {a.val};\n')

            elif a.type.type == b.type.type == ValueClsType.PTR:
                sb.add(f'{'':{indent}}{b.val} = {a.val};\n')

            else:
                unreachable(instr)

    sb_global = StringBuilder()
    sb_global += f'#include <stdio.h>\n'
    sb_global += f'#include <stdbool.h>\n'
    sb_global += f'#include <stdlib.h>\n'
    sb_global += f'#include <stdint.h>\n'

    for module, proc in ((module, proc) for module in ir.modules for proc in module.procs):
        sb = StringBuilder()
        sb_header = StringBuilder()
        sb_struct = StringBuilder()

        instrs = proc.instrs
        stack: Stack = []
        bound_vars: list[tuple[str, StackEntry]] = []

        proc_name = c_proc_ident(module.id, proc.name)
        ret = f'ret_{proc_name}'
        sb_header += f'{ret} proc_{proc_name}('
        for i, typ in enumerate(proc.contract.ins):
            name = get_varname('arg')
            if i > 0:
                sb_header += ', '
            sb_header += f'{c_type(typ)} {name}'
            stack.append(StackEntry(typ, name, proc.tok))
        sb_header += ') {\n'
        sb_struct += f'typedef struct {{\n'
        for i, typ in enumerate(proc.contract.outs):
            name = f'_{i}'
            sb_struct += f'  {c_type(typ)} {name};\n'
        sb_struct += f'}} {ret};\n'
        indent = 2

        block_stack: list[Block] = []
        for ip, instr in enumerate(instrs):
            ip += 1
            match instr.type:
                case InstructionType.PUSH_INT:
                    var = get_varname('lit_int')
                    typ = ValueCls(ValueClsType.INT)
                    declare_var(var, typ)
                    stack.append(StackEntry(typ, var, tok=instr.tok))
                    sb += f'{'':{indent}}{var} = {instr.arg1};\n'

                case InstructionType.PUSH_BOOL:
                    var = get_varname('lit_bool')
                    typ = ValueCls(ValueClsType.BOOL)
                    declare_var(var, typ)
                    stack.append(StackEntry(typ, var, tok=instr.tok))
                    sb += f'{'':{indent}}{var} = {"true" if instr.tok.value else "false"};\n'

                case InstructionType.PUSH_TYPE:
                    typ = ValueCls(ValueClsType.TYPE, instr.arg1)
                    stack.append(StackEntry(typ, unused, tok=instr.tok))

                case InstructionType.WORD:
                    proc_ref = cast(ProcRef, instr.arg1)
                    proc_called = ir.get_proc_by_ref(proc_ref)
                    if proc_called is None:
                        unreachable(instr, f'proc {proc_ref.name} not found')

                    proc_called_name = c_proc_ident(proc_ref.module, proc_called.name)
                    ret_var = get_varname(f'res_{proc_called_name}')
                    ret_type = f'ret_{proc_called_name}'
                    sb += f'{'':{indent}}{ret_type} {ret_var} = proc_{proc_called_name}('
                    for i, arg in enumerate(stack[len(stack) - len(proc_called.contract.ins) :]):
                        if i > 0:
                            sb += ', '
                        sb += f'{arg.val}'
                        _ = stack.pop()
                    sb += ');\n'

                    for i, out in enumerate(proc_called.contract.outs):
                        typ = out
                        var = get_varname(f'res_{proc_called_name}_{i}')
                        declare_var(var, typ)
                        stack.append(StackEntry(typ, var, tok=instr.tok))
                        sb += f'{'':{indent}}{var} = {ret_var}._{i};\n'

                case InstructionType.IF:
                    block = Block(InstructionType.IF)
                    block_stack.append(block)

                case InstructionType.ELSE:
                    block = block_stack[-1]
                    if block.type != InstructionType.IF:
                        unreachable(instr)
                    if block.stack is None:
                        unreachable(instr)

                    copy_stacks(stack, block.stack)
                    stack_copy = stack.copy()
                    stack = block.stack
                    block.stack = stack_copy
                    indent -= 2
                    sb += f'{'':{indent}}}} else {{\n'
                    indent += 2

                case InstructionType.END:
                    block = block_stack.pop()
                    if block.type == InstructionType.IF:
                        if block.stack is None:
                            unreachable(instr)

                        copy_stacks(stack, block.stack)
                        stack = block.stack
                        indent -= 2
                        sb += f'{'':{indent}}}}\n'

                    elif block.type == InstructionType.WHILE:
                        if block.stack is None:
                            unreachable(instr)

                        copy_stacks(stack, block.stack)
                        stack = block.stack
                        indent -= 2
                        sb += f'{'':{indent}}}} else break;\n'
                        indent -= 2
                        sb += f'{'':{indent}}}}\n'

                    elif block.type == InstructionType.PROC:
                        indent -= 2
                        sb += f'{'':{indent}}}}\n'

                    else:
                        unreachable(instr)

                case InstructionType.WHILE:
                    block = Block(InstructionType.WHILE)
                    block_stack.append(block)
                    sb += f'{'':{indent}}while (true) {{\n'
                    indent += 2

                case InstructionType.DO:
                    a = stack.pop()
                    if a.type.type == ValueClsType.BOOL:
                        var = a.val

                        block = block_stack[-1]
                        if block.type == InstructionType.IF:
                            sb += f'{'':{indent}}if ({var}) {{\n'
                            indent += 2
                            stack_copy = stack.copy()
                            block.stack = stack_copy

                        elif block.type == InstructionType.WHILE:
                            sb += f'{'':{indent}}if ({var}) {{\n'
                            indent += 2
                            stack_copy = stack.copy()
                            block.stack = stack_copy

                        else:
                            unreachable(instr)

                    else:
                        unreachable(instr, stack=stack)

                case InstructionType.PROC:
                    unreachable(instr, 'proc instrs must not me emitted')

                case InstructionType.RET:
                    ret_type = f'ret_{proc_name}'
                    sb += f'{'':{indent}}return ({ret_type}){{\n'
                    indent += 2
                    for i, x in enumerate(stack):
                        sb += f'{'':{indent}}._{i} = {x.val},\n'
                    indent -= 2
                    sb += f'{'':{indent}}}};\n'
                    stack = []

                case InstructionType.LABEL:
                    notimplemented(instr, f'translating {InstructionType.LABEL}')

                case InstructionType.BIND:
                    bound_vars.append((instr.arg1, stack.pop()))

                case InstructionType.UNBIND:
                    _ = bound_vars.pop()

                case InstructionType.LOAD_BIND:
                    for name, value in reversed(bound_vars):
                        if name == instr.arg1:
                            stack.append(value)
                            break

                case InstructionType.INTRINSIC:
                    intr_type = cast(IntrinsicType, instr.arg1)
                    match intr_type:
                        # arithmetic:
                        case IntrinsicType.ADD:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'add')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} + {b.val};\n'

                            elif a.type.type == ValueClsType.PTR and b.type.type == ValueClsType.INT:
                                var = get_varname(f'ptr_add')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = (void*)((char*){a.val} + {b.val});\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.SUB:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'sub')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} - {b.val};\n'

                            elif a.type.type == ValueClsType.PTR and b.type.type == ValueClsType.INT:
                                var = get_varname(f'ptr_sub')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = (void*)((char*){a.val} - {b.val});\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.MUL:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'mul')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} * {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.DIV:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'div')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                emit_division_by_zero_check(instr, a, b)
                                sb += f'{'':{indent}}{var} = {a.val} / {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.MOD:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'mod')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                emit_division_by_zero_check(instr, a, b)
                                sb += f'{'':{indent}}{var} = {a.val} % {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.DIVMOD:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var1 = get_varname(f'div')
                                var2 = get_varname(f'mod')
                                declare_var(var1, a.type)
                                declare_var(var2, a.type)
                                stack.append(StackEntry(a.type, var1, tok=instr.tok))
                                stack.append(StackEntry(a.type, var2, tok=instr.tok))
                                emit_division_by_zero_check(instr, a, b)
                                sb += f'{'':{indent}}{var1} = {a.val} / {b.val};\n'
                                sb += f'{'':{indent}}{var2} = {a.val} % {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.SHL:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'shl')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} << {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.SHR:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'shr')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} >> {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.BOR:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'bor')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} | {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.BAND:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'band')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} & {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.BXOR:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'bxor')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} ^ {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.BNOT:
                            # a -- ~a
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == ValueClsType.INT:
                                var = get_varname(f'bnot')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = ~{a.val};\n'

                            else:
                                unreachable(instr)

                        # logic:
                        case IntrinsicType.AND:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == b.type.type == ValueClsType.BOOL:
                                var = get_varname(f'and')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} && {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.OR:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == b.type.type == ValueClsType.BOOL:
                                var = get_varname(f'or')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} || {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.NOT:
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == ValueClsType.BOOL:
                                var = get_varname(f'not')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = !{a.val};\n'

                            else:
                                unreachable(instr)

                        # comparison:
                        case IntrinsicType.EQ:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'eq')
                                declare_var(var, a.type)
                                stack.append(StackEntry(ValueCls(ValueClsType.BOOL), var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} == {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.NE:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'ne')
                                declare_var(var, a.type)
                                stack.append(StackEntry(ValueCls(ValueClsType.BOOL), var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} != {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.LT:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'lt')
                                declare_var(var, a.type)
                                stack.append(StackEntry(ValueCls(ValueClsType.BOOL), var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} < {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.GT:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'gt')
                                declare_var(var, a.type)
                                stack.append(StackEntry(ValueCls(ValueClsType.BOOL), var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} > {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.LE:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'le')
                                declare_var(var, a.type)
                                stack.append(StackEntry(ValueCls(ValueClsType.BOOL), var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} <= {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.GE:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 4)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'ge')
                                declare_var(var, a.type)
                                stack.append(StackEntry(ValueCls(ValueClsType.BOOL), var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} >= {b.val};\n'

                            else:
                                unreachable(instr)

                        # stack manipulation:
                        case IntrinsicType.DUP:
                            a = stack.pop()
                            stack.append(a)

                            var = get_varname(f'dup')
                            declare_var(var, a.type)
                            stack.append(StackEntry(a.type, var, tok=instr.tok))
                            sb += f'{'':{indent}}{var} = {a.val};\n'

                        case IntrinsicType.DUP2:
                            b = stack.pop()
                            a = stack.pop()
                            stack.append(a)
                            stack.append(b)

                            var_a = get_varname(f'dup2')
                            declare_var(var_a, a.type)
                            stack.append(StackEntry(a.type, var_a, tok=instr.tok))
                            sb += f'{'':{indent}}{var_a} = {a.val};\n'

                            var_b = get_varname(f'dup2')
                            declare_var(var_b, b.type)
                            stack.append(StackEntry(b.type, var_b, tok=instr.tok))
                            sb += f'{'':{indent}}{var_b} = {b.val};\n'

                        case IntrinsicType.DROP:
                            _ = stack.pop()

                        case IntrinsicType.SWAP:
                            b = stack.pop()
                            a = stack.pop()
                            stack.append(b)
                            stack.append(a)

                        case IntrinsicType.SWAP2:
                            y = stack.pop()
                            x = stack.pop()
                            b = stack.pop()
                            a = stack.pop()
                            stack.append(x)
                            stack.append(y)
                            stack.append(a)
                            stack.append(b)

                        case IntrinsicType.ROT:
                            c = stack.pop()
                            b = stack.pop()
                            a = stack.pop()
                            stack.append(b)
                            stack.append(c)
                            stack.append(a)

                        case IntrinsicType.CAST:
                            # x T -- (T)x
                            if len(stack) < 2:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects two items on the stack',
                                    stack=stack,
                                )
                            t = stack.pop()
                            x = stack.pop()
                            if t.type.type != ValueClsType.TYPE:
                                error(
                                    instr,
                                    f'cast expects a type to cast to, but got {pp(t.type)}',
                                    stack=stack,
                                )

                            trace(
                                instr,
                                'casting during codegen',
                                x=x,
                                t=t,
                            )
                            match (x.type, t.type.val):
                                case _ if x.type == t.type.val:
                                    warn(
                                        instr,
                                        f'redundant cast {pp(x.type)} to {pp(t.type.val)}',
                                    )
                                    stack.append(StackEntry(t.type.val, x.val, tok=instr.tok))

                                case (
                                    (ValueCls(ValueClsType.INT), ValueCls(ValueClsType.BOOL))
                                    | (ValueCls(ValueClsType.BOOL), ValueCls(ValueClsType.INT))
                                    | (ValueCls(ValueClsType.INT), ValueCls(ValueClsType.PTR))
                                    | (ValueCls(ValueClsType.PTR), ValueCls(ValueClsType.INT))
                                ):
                                    var = get_varname(f'cast')
                                    declare_var(var, type=t.type.val)
                                    stack.append(StackEntry(type=t.type.val, val=var, tok=instr.tok))
                                    sb += f'{'':{indent}}{var} = ({c_type(t.type.val)}){x.val};\n'

                                case _:
                                    error(
                                        instr,
                                        f'cannot cast {pp(x.type)} to {pp(t.type.val)}',
                                        stack=stack,
                                    )

                        # memory:
                        case IntrinsicType.ALLOC:
                            size = stack.pop()
                            var = get_varname(f'alloc')
                            typ = ValueCls(ValueClsType.PTR)
                            declare_var(var, typ)
                            stack.append(StackEntry(typ, var, tok=instr.tok))
                            sb += f'{'':{indent}}{var} = malloc({size.val});\n'
                            sb += f'{'':{indent}}if ({var} == NULL) {{ printf("[ERROR] allocation failed\\n"); exit(1); }}\n'

                        case (
                            IntrinsicType.READ_I8
                            | IntrinsicType.READ_I16
                            | IntrinsicType.READ_I32
                            | IntrinsicType.READ_I64
                            | IntrinsicType.READ_U8
                            | IntrinsicType.READ_U16
                            | IntrinsicType.READ_U32
                            | IntrinsicType.READ_U64
                        ):
                            ptr = stack.pop()
                            bits, signed = MEMORY_ACCESS_BITS[intr_type]
                            c_read_type = f'int{bits}_t' if signed else f'uint{bits}_t'
                            var = get_varname(f'read')
                            typ = ValueCls(ValueClsType.INT)
                            declare_var(var, typ)
                            stack.append(StackEntry(typ, var, tok=instr.tok))
                            sb += f'{'':{indent}}{var} = (int)(*({c_read_type}*){ptr.val});\n'

                        case (
                            IntrinsicType.WRITE_I8
                            | IntrinsicType.WRITE_I16
                            | IntrinsicType.WRITE_I32
                            | IntrinsicType.WRITE_I64
                            | IntrinsicType.WRITE_U8
                            | IntrinsicType.WRITE_U16
                            | IntrinsicType.WRITE_U32
                            | IntrinsicType.WRITE_U64
                        ):
                            ptr = stack.pop()
                            val = stack.pop()
                            bits, signed = MEMORY_ACCESS_BITS[intr_type]
                            c_write_type = f'int{bits}_t' if signed else f'uint{bits}_t'
                            sb += f'{'':{indent}}*({c_write_type}*){ptr.val} = ({c_write_type}){val.val};\n'

                        # debugging:
                        case IntrinsicType.PRINT:
                            a = stack.pop()
                            if a.type.type == ValueClsType.INT:
                                sb += f'{'':{indent}}printf("%d\\n", {a.val});\n'

                            elif a.type.type == ValueClsType.BOOL:
                                sb += f'{'':{indent}}printf("%s\\n", {a.val} ? "true" : "false");\n'

                            else:
                                notimplemented(instr, f'printing {pp(a.type)} is not implemented yet')

                        case IntrinsicType.PUTC:
                            a = stack.pop()
                            if a.type.type == ValueClsType.INT:
                                sb += f'{'':{indent}}putchar({a.val});\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.DEBUG:
                            pass

                        case _:
                            assert_never(intr_type)
                case _:
                    assert_never(instr.type)

        if stack:
            unreachable(None, stack=stack)

        if block_stack:
            unreachable(None, block_stack=block_stack)

        sb += '}\n'
        sb_global += str(sb_struct)
        sb_global += str(sb_header)
        sb_global += str(sb)

    sb_global += 'int main() {\n'
    sb_global += '  proc_main();\n'
    sb_global += '  return 0;\n'
    sb_global += '}\n'

    code = str(sb_global)

    trace(
        loc_unknown,
        'Generated C code:',
        generated_code=code,
    )
    return code
