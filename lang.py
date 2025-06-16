from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Never, assert_never, cast, final, override
from enum import Enum, auto
from dataclasses import dataclass, field, is_dataclass
import traceback
import sys

from sb import StringBuilder


@final
class _sentinel:
    def __init__(self, name: str) -> None:
        self.name = name

    @override
    def __repr__(self) -> str:
        return f'<{self.name}>'


missing = _sentinel('missing')
unused = _sentinel('unused')
typechecking = _sentinel('typechecking')


@dataclass(frozen=True)
class Loc:
    file: str
    line: int
    col: int

    @override
    def __repr__(self) -> str:
        return f'{self.file}:{self.line}:{self.col}'


loc_unknown = Loc('<?>', 0, 0)


class TokenType(Enum):
    INT = auto()
    BOOL = auto()
    CHAR = auto()
    STR = auto()
    # FLOAT = auto()
    WORD = auto()
    KEYWORD = auto()

    IMAGINARY = auto()


class KeywordType(Enum):
    PROC = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    DO = auto()
    END = auto()
    TYPE_DELIM = auto()  # --


@dataclass
class Token:
    type: TokenType
    value: Any
    loc: Loc = field(compare=False)


class InstructionType(Enum):
    PUSH_INT = auto()
    PUSH_BOOL = auto()
    WORD = auto()
    INTRINSIC = auto()

    PROC = auto()
    IF = auto()  # cosmetic
    ELSE = auto()  # jumps unconditionally to the end
    WHILE = auto()  # cosmetic
    DO = auto()  # jumps conditionally to the end of loop or to the else branch
    END = auto()  # jumps unconditionally to the beginning of loop or after the if
    LABEL = auto()
    RET = auto()


class IntrinsicType(Enum):
    # arithhmetic:
    ADD = auto()  # a b -- a+b
    SUB = auto()  # a b -- a-b
    MUL = auto()  # a b -- a*b
    DIV = auto()  # a b -- a/b
    MOD = auto()  # a b -- a%b
    DIVMOD = auto()  # a b -- a/b a%b

    SHL = auto()  # a b -- a<<b
    SHR = auto()  # a b -- a>>b

    # bitwise:
    BOR = auto()  # a b -- a|b
    BAND = auto()  # a b -- a&b
    BXOR = auto()  # a b -- a^b
    BNOT = auto()  # a -- ~a

    # logic:
    AND = auto()  # a b -- (a and b)
    OR = auto()  # a b -- (a or b)
    NOT = auto()  # a -- (not a)

    # comparison:
    EQ = auto()  # a b -- a==b
    NE = auto()  # a b -- a!=b
    LT = auto()  # a b -- a<b
    GT = auto()  # a b -- a>b
    LE = auto()  # a b -- a<=b
    GE = auto()  # a b -- a>=b

    # stack manipulation:
    DUP = auto()  # a -- a a
    DROP = auto()  # a --
    SWAP = auto()  # a b -- b a
    ROT = auto()  # a b c -- b c a

    # debugging:
    PRINT = auto()  # a --
    DEBUG = auto()  # --


@dataclass
class Instruction:
    type: InstructionType
    tok: Token = field(compare=False)
    arg1: Any = unused
    arg2: Any = unused


@dataclass
class Proc:
    name: str
    ins: list[ValueCls]
    outs: list[ValueCls]
    instrs: list[Instruction] = field(default_factory=list)


@dataclass
class IR:
    procs: list[Proc] = field(default_factory=list)

    def get_proc_by_name(self, name: str) -> Proc | None:
        for proc in self.procs:
            if proc.name == name:
                return proc
        return None


class ValueClsType(Enum):
    INT = auto()
    BOOL = auto()


@dataclass
class ValueCls:
    type: ValueClsType
    val: Any
    tok: Token = field(compare=False)


@dataclass
class StackEntry:
    type: ValueCls
    val: Any
    tok: Token = field(compare=False)


class Error(Exception): ...


type Stack = list[StackEntry]


def expect_enum_size(e_cls: type[Enum], expected_size: int) -> None:
    if len(e_cls) != expected_size:
        raise Error(f'{e_cls.__name__} has {len(e_cls)} members, expected {expected_size}')


type _Locatable = Instruction | Token | Loc | StackEntry | None


def pp(x: Any) -> str:
    match x:
        case Enum():
            return x.name

        case StackEntry():
            if x.val is typechecking:
                return f'{pp(x.type)}@{pp(x.tok.loc)}'
            return f'{pp(x.type)}:{pp(x.val)}@{pp(x.tok.loc)}'

        case ValueCls():
            match x.type:
                case ValueClsType.INT:
                    return 'INT'
                case ValueClsType.BOOL:
                    return 'BOOL'
                case _:
                    assert_never(x.type)

        case Token():
            return f'{x.loc}:{pp(x.type)}:{pp(x.value)}'

        case IR():
            sb = StringBuilder()
            sb += 'IR:\n'
            for proc in x.procs:
                sb += f'proc {proc.name}:\n'
                for i, instr in enumerate(proc.instrs):
                    sb += f'{i:3}. {pp(instr)}\n'

            return str(sb)

        case Instruction():
            s = f'{x.type.name}'
            if x.arg1 is not unused:
                s += f':{pp(x.arg1)}'
            if x.arg2 is not unused:
                s += f':{pp(x.arg2)}'
            return s

        case _ if is_dataclass(x):
            res = repr(x)
            cls = type(x)
            res = res.replace(cls.__qualname__, cls.__name__)
            return res

        case _:
            return repr(x)


def _locatable_to_loc(loc: _Locatable) -> Loc:
    match loc:
        case Loc():
            return loc
        case Token():
            return loc.loc
        case Instruction():
            return loc.tok.loc
        case StackEntry():
            return loc.tok.loc
        case None:
            return loc_unknown
        case _:
            assert_never(loc)


def note(**notes: Any) -> None:
    for k, v in notes.items():
        print(f'[NOTE] {k}:', end='')
        match v:
            case list():
                v = cast(list[Any], v)
                if not v:
                    print(f' (empty)')

                else:
                    print()
                    for i, x in enumerate(v):
                        print(f'    {i:3}. {pp(x)}')

            case _:
                if not isinstance(v, str):
                    v = pp(v)
                if '\n' not in v:
                    print(f' {v}')

                else:
                    print()
                    for line in v.splitlines():
                        print(f'    {line}')


LL_ERROR = 0
LL_WARN = 1
LL_INFO = 2
LL_TRACE = 3

LL_DEFAULT = LL_INFO
log_level = LL_INFO

LL = {
    'ERROR': LL_ERROR,
    'WARN': LL_WARN,
    'INFO': LL_INFO,
    'TRACE': LL_TRACE,
    'DEFAULT': LL_DEFAULT,
}


def error(loc: _Locatable, msg: str, exitcode: int = 1, **notes: Any) -> Never:
    if log_level >= LL_ERROR:
        msg = f'[ERROR] {_locatable_to_loc(loc)}: {msg}'
        print(msg)
        note(**notes)
    sys.exit(exitcode)


def warn(loc: _Locatable, msg: str, **notes: Any) -> None:
    if log_level >= LL_WARN:
        msg = f'[WARN] {_locatable_to_loc(loc)}: {msg}'
        print(msg)
        note(**notes)


def info(loc: _Locatable, msg: str, **notes: Any) -> None:
    if log_level >= LL_INFO:
        print(f'[INFO] {_locatable_to_loc(loc)}: {msg}')
        note(**notes)


def trace(loc: _Locatable, msg: str, **notes: Any) -> None:
    if log_level >= LL_TRACE:
        print(f'[TRACE] {_locatable_to_loc(loc)}: {msg}')
        note(**notes)


def typecheck_has_a_bug(loc: _Locatable, msg: str, **notes: Any) -> Never:
    error(
        loc,
        f'this should not happen because this must have been caugth at typecheck time: {msg}',
        **notes,
    )


def unreachable(loc: _Locatable, msg: str = '<?>', **notes: Any) -> Never:
    traceback.print_stack(file=sys.stdout)
    error(loc, f'unreachable: {msg}', **notes)


def notimplemented(loc: _Locatable, msg: str) -> Never:
    error(loc, f'not implemented: {msg}')


def tokenize(text: str, filename: str = '<?>') -> Iterable[Token]:
    def is_int(s: str) -> bool:
        return set(s) <= set('0123456789')

    line_no = 1
    col_no = 0

    loc_start = Loc(filename, line_no, col_no)
    i_start = 0

    text += '\n'

    i = 0
    while i < len(text):
        c = text[i]

        col_no += 1
        if c == '\n':
            line_no += 1
            col_no = 0

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
                match chunk:
                    case 'proc':
                        yield Token(TokenType.KEYWORD, KeywordType.PROC, loc_start)
                    case 'if':
                        yield Token(TokenType.KEYWORD, KeywordType.IF, loc_start)
                    case 'else':
                        yield Token(TokenType.KEYWORD, KeywordType.ELSE, loc_start)
                    case 'end':
                        yield Token(TokenType.KEYWORD, KeywordType.END, loc_start)
                    case 'while':
                        yield Token(TokenType.KEYWORD, KeywordType.WHILE, loc_start)
                    case 'do':
                        yield Token(TokenType.KEYWORD, KeywordType.DO, loc_start)
                    case '--':
                        yield Token(TokenType.KEYWORD, KeywordType.TYPE_DELIM, loc_start)
                    case _:
                        yield Token(TokenType.WORD, chunk, loc_start)

            loc_start = Loc(filename, line_no, col_no)
            i_start = i + 1

        i += 1


def compile(toks: list[Token]) -> IR:
    ir = IR()

    implicit_main = False
    cur_proc: Proc | None = None

    def add_instr(instr: Instruction) -> int:
        """returns index of added instruction"""
        nonlocal cur_proc
        nonlocal implicit_main
        if cur_proc is None:
            cur_proc = ir.get_proc_by_name('main')
            implicit_main = True
        if cur_proc is None:
            cur_proc = Proc(name='main', ins=[], outs=[])
            ir.procs.append(cur_proc)
        cur_proc.instrs.append(instr)
        return len(cur_proc.instrs) - 1

    # pyright tries to be smart and infer that `cur_proc` is always `None`
    # trick it into thinking that it might be non-`None`
    if len('abc') == 4:
        cur_proc = Proc(name='lol', ins=[], outs=[])

    # _label_cnt = 0
    # def get_label() -> int:
    #     nonlocal _label_cnt
    #     _label_cnt += 1
    #     return _label_cnt

    @dataclass
    class Block:
        type: InstructionType
        ip1: int = -1
        ip2: int = -1
        ip3: int = -1
        ip4: int = -1

    block_stack: list[Block] = []

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
                        b = Block(
                            InstructionType.IF,
                        )
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
                        b = Block(
                            InstructionType.WHILE,
                        )
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

                    case KeywordType.END:
                        if not block_stack:
                            error(tok, f'END should follow an IF or WHILE')
                        b = block_stack.pop()

                        match b.type:
                            # if // ip1
                            #   <cond>
                            # do // ip2
                            #   <body>
                            # else // ip3
                            #   <body>
                            # end // ip4
                            case InstructionType.IF:
                                if b.ip1 == -1:
                                    unreachable(
                                        tok,
                                        f'somehow the IF doesnt have an IF-instruction address saved',
                                    )
                                if b.ip2 == -1:
                                    error(tok, f'if <cond> do <body> [else <body>] end')

                                if b.ip3 == -1:
                                    b.ip3 = add_instr(Instruction(type=InstructionType.ELSE, tok=tok, arg1=missing))

                                assert cur_proc is not None
                                b.ip4 = add_instr(Instruction(type=InstructionType.END, tok=tok, arg1=-1))
                                cur_proc.instrs[-1].arg1 = b.ip4 + 1

                                cur_proc.instrs[b.ip2].arg1 = b.ip3
                                cur_proc.instrs[b.ip3].arg1 = b.ip4

                            # while // ip1
                            #   <cond>
                            # do // ip2
                            #   <body>
                            # end // ip3
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

                            case _:
                                error(tok, f'END should follow an IF or WHILE, not {pp(b.type)}')

                    case KeywordType.TYPE_DELIM:
                        error(tok, f'{kw_type} should not be used only in proc signatures')

                    case KeywordType.PROC:
                        if cur_proc is not None:
                            error(tok, f'PROC should not be inside of another PROC: {cur_proc.name}')

                        if i >= len(toks):
                            error(tok, f'expected a name after PROC')
                        tok = toks[i]
                        if tok.type != TokenType.WORD:
                            error(tok, f'expected a name after PROC, got {tok.type}')
                        name = tok.value
                        i += 1

                        def parse_type(tok: Token) -> ValueCls:
                            if tok.type != TokenType.WORD:
                                error(tok, f'expected a type after proc name, got {tok.type}')
                            match tok.value:
                                case 'int':
                                    return ValueCls(type=ValueClsType.INT, val=unused, tok=tok)
                                case 'bool':
                                    return ValueCls(type=ValueClsType.BOOL, val=unused, tok=tok)
                                case _:
                                    error(tok, f'expected a type after proc name, got {tok.value}')

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

                        cur_proc = Proc(name=name, ins=ins, outs=outs)
                        ir.procs.append(cur_proc)
                        block = Block(InstructionType.PROC)
                        block_stack.append(block)

                    case _:
                        assert_never(kw_type)

            case TokenType.WORD:
                expect_enum_size(IntrinsicType, 27)
                match tok.value:
                    # arithmetic:
                    case '+':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.ADD))
                    case '-':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.SUB))
                    case '*':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.MUL))
                    case '/':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.DIV))
                    case '%':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.MOD))
                    case '/%':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.DIVMOD))

                    case '<<':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.SHL))
                    case '>>':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.SHR))

                    # bitwise:
                    case '&':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.BAND))
                    case '|':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.BOR))
                    case '^':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.BXOR))
                    case '~':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.BNOT))

                    # logic:
                    case '&&':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.AND))
                    case '||':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.OR))
                    case '!':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.NOT))

                    # comparison:
                    case '==':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.EQ))
                    case '!=':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.NE))
                    case '<':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.LT))
                    case '>':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.GT))
                    case '<=':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.LE))
                    case '>=':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.GE))

                    # stack manipulation:
                    case 'dup':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.DUP))
                    case 'drop':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.DROP))
                    case 'swap':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.SWAP))
                    case 'rot':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.ROT))

                    # debugging:
                    case 'print':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.PRINT))
                    case '?':
                        _ = add_instr(Instruction(type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.DEBUG))

                    case _:
                        _ = add_instr(Instruction(type=InstructionType.WORD, tok=tok, arg1=tok.value))

            case _:
                assert_never(tok.type)

    if implicit_main:
        _ = add_instr(Instruction(type=InstructionType.RET, tok=Token(TokenType.IMAGINARY, '', loc_unknown)))

    if block_stack:
        error(
            None,
            'unclosed blocks',
            blocks=block_stack,
        )

    if not ir.get_proc_by_name('main'):
        ir.procs.append(Proc(name='main', ins=[], outs=[]))

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

    # TODO
    main = ir.get_proc_by_name('main')
    assert main is not None
    block_stack: list[Block] = []
    stack: Stack = []

    for proc in ir.procs:
        stack = []
        for type_in in proc.ins:
            stack.append(StackEntry(type_in, typechecking, tok=type_in.tok))
        instrs = proc.instrs
        for instr in instrs:
            match instr.type:
                case InstructionType.PUSH_INT:
                    stack.append(StackEntry(ValueCls(ValueClsType.INT, unused, instr.tok), typechecking, tok=instr.tok))

                case InstructionType.PUSH_BOOL:
                    stack.append(
                        StackEntry(ValueCls(ValueClsType.BOOL, unused, instr.tok), typechecking, tok=instr.tok)
                    )

                case InstructionType.WORD:
                    proc_called = ir.get_proc_by_name(instr.arg1)
                    if proc_called is None:
                        error(instr, f'unknown word {instr.arg1}')

                    # 1. check that stack and proc_called.ins are the same
                    if len(stack) < len(proc_called.ins):
                        error(
                            instr,
                            f'stack too small for {instr.arg1}: expected {len(proc_called.ins)} but got {len(stack)}',
                            stack=stack,
                            ins=proc_called.ins,
                        )

                    for i, (e1, e2) in enumerate(zip(reversed(stack), proc_called.ins)):
                        if e1.type != e2:
                            error(
                                instr,
                                f'stack doesnt match at {i} for {instr.arg1}: expected {pp(e1.type)} but got {pp(e2)}',
                                stack=stack,
                                ins=proc_called.ins,
                            )

                    # 2. remove proc_called.ins from stack
                    for _ in proc_called.ins:
                        _ = stack.pop()

                    # 3. put proc_called.outs on stack
                    for out in proc_called.outs:
                        stack.append(StackEntry(out, typechecking, tok=instr.tok))

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
                    # check that the stack contains correct return types
                    # x: ValueCls = stack[i].val
                    # outs: list[ValueCls]
                    if len(stack) != len(proc.outs):
                        error(
                            instr,
                            f'return type mismatch: expected {len(proc.outs)} items on the stack, got {len(stack)}',
                            stack=stack,
                            outs=proc.outs,
                        )

                    for i, (item_stack, item_out) in enumerate(zip(stack, proc.outs)):
                        t1 = item_stack.type
                        t2 = item_out
                        if t1 != t2:
                            error(
                                instr,
                                f'return type mismatch at {i}: expected {t2} on the stack, got {t1}',
                                stack=stack,
                                outs=proc.outs,
                            )

                case InstructionType.LABEL:
                    notimplemented(instr, f'typechecking {InstructionType.LABEL}')

                case InstructionType.INTRINSIC:
                    intr_type = cast(IntrinsicType, instr.arg1)
                    expect_enum_size(IntrinsicType, 27)
                    match intr_type:
                        # arithmetic:
                        case IntrinsicType.ADD | IntrinsicType.SUB:
                            # a b -- a+b
                            if len(stack) < 2:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects two INTs on the stack',
                                    stack=stack,
                                )
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 2)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                stack.append(StackEntry(a.type, typechecking, tok=instr.tok))

                            else:
                                error(
                                    instr,
                                    f'cannot add/subtract {pp(a.type)} and {pp(b.type)}',
                                    other_stack_items=stack,
                                )

                        case IntrinsicType.MUL | IntrinsicType.DIV:
                            # a b -- a*b
                            if len(stack) < 2:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects two INTs on the stack',
                                    other_stack_items=stack,
                                )
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 2)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                stack.append(StackEntry(a.type, typechecking, tok=instr.tok))

                            else:
                                error(
                                    instr,
                                    f'cannot multiply/divide {pp(a.type)} by {pp(b.type)}',
                                    stack=stack,
                                )

                        case IntrinsicType.MOD:
                            # a b -- a%b
                            if len(stack) < 2:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects two INTs on the stack',
                                    stack=stack,
                                )
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 2)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                stack.append(StackEntry(a.type, typechecking, tok=instr.tok))

                            else:
                                error(
                                    instr,
                                    f'cannot modulo {pp(a.type)} by {pp(b.type)}',
                                    stack=stack,
                                )

                        case IntrinsicType.DIVMOD:
                            # a b -- a//b a%b
                            if len(stack) < 2:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects two INTs on the stack',
                                    stack=stack,
                                )
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 2)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                stack.append(StackEntry(a.type, typechecking, tok=instr.tok))
                                stack.append(StackEntry(a.type, typechecking, tok=instr.tok))

                            else:
                                error(
                                    instr,
                                    f'cannot divmod {pp(a.type)} by {pp(b.type)}',
                                    stack=stack,
                                )

                        case IntrinsicType.SHL | IntrinsicType.SHR:
                            # a b -- a@@b
                            if len(stack) < 2:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects two INTs on the stack',
                                    stack=stack,
                                )
                            b = stack.pop()
                            a = stack.pop()
                            if a.type.type == b.type.type == ValueClsType.INT:
                                stack.append(StackEntry(a.type, typechecking, tok=instr.tok))

                            else:
                                error(
                                    instr,
                                    f'cannot shift {pp(a.type)} by {pp(b.type)}',
                                    stack=stack,
                                )

                        case IntrinsicType.BOR | IntrinsicType.BAND | IntrinsicType.BXOR:
                            # a b -- a@b
                            if len(stack) < 2:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects two INTs on the stack',
                                    stack=stack,
                                )
                            b = stack.pop()
                            a = stack.pop()
                            if a.type.type == b.type.type == ValueClsType.INT:
                                stack.append(StackEntry(a.type, typechecking, tok=instr.tok))

                            else:
                                error(
                                    instr,
                                    f'cannot bitwise or/and/xor {pp(a.type)} with {pp(b.type)}',
                                    stack=stack,
                                )

                        case IntrinsicType.BNOT:
                            # a -- ~a
                            if len(stack) < 1:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects one INT on the stack',
                                    stack=stack,
                                )
                            a = stack.pop()
                            if a.type.type == ValueClsType.INT:
                                stack.append(StackEntry(a.type, typechecking, tok=instr.tok))

                            else:
                                error(
                                    instr,
                                    f'cannot bitwise not an {pp(a.type)}',
                                    stack=stack,
                                )

                        # logic:
                        case IntrinsicType.AND | IntrinsicType.OR:
                            # a b -- (a op b)
                            if len(stack) < 2:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects two BOOLs on the stack',
                                    stack=stack,
                                )
                            b = stack.pop()
                            a = stack.pop()
                            if a.type.type == b.type.type == ValueClsType.BOOL:
                                stack.append(StackEntry(a.type, typechecking, tok=instr.tok))

                            else:
                                error(
                                    instr,
                                    f'cannot and/or {pp(a.type)} with {pp(b.type)}',
                                    stack=stack,
                                )

                        case IntrinsicType.NOT:
                            # a -- (not a)
                            if len(stack) < 1:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects one BOOL on the stack',
                                    stack=stack,
                                )
                            a = stack.pop()
                            if a.type.type == ValueClsType.BOOL:
                                stack.append(StackEntry(a.type, typechecking, tok=instr.tok))

                            else:
                                error(
                                    instr,
                                    f'cannot not {pp(a.type)}',
                                    stack=stack,
                                )

                        # comparison:
                        case IntrinsicType.EQ | IntrinsicType.NE:
                            # a b -- (a op b)
                            if len(stack) < 2:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects two INTs on the stack',
                                    stack=stack,
                                )
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 2)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                stack.append(
                                    StackEntry(
                                        ValueCls(ValueClsType.BOOL, unused, instr.tok), typechecking, tok=instr.tok
                                    )
                                )

                            else:
                                error(
                                    instr,
                                    f'cannot compare {pp(a.type)} and {pp(b.type)}',
                                    stack=stack,
                                )

                        case IntrinsicType.LT | IntrinsicType.LE | IntrinsicType.GT | IntrinsicType.GE:
                            # a b -- (a op b)
                            if len(stack) < 2:
                                error(
                                    instr,
                                    f'not enough items on stack for {intr_type}: it expects two INTs on the stack',
                                    stack=stack,
                                )
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 2)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                stack.append(
                                    StackEntry(
                                        ValueCls(ValueClsType.BOOL, unused, instr.tok), typechecking, tok=instr.tok
                                    )
                                )

                            else:
                                error(
                                    instr,
                                    f'cannot compare {pp(a.type)} and {pp(b.type)}',
                                    stack=stack,
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

    main = ir.get_proc_by_name('main')
    assert main is not None

    stack: list[StackEntry] = []
    frame_stack: list[Frame] = [Frame(main.instrs)]

    while frame_stack and frame_stack[-1].ip < len(frame_stack[-1].instrs):
        frame = frame_stack[-1]
        instr = frame.instrs[frame.ip]
        frame.ip += 1
        match instr.type:
            case InstructionType.PUSH_INT:
                stack.append(StackEntry(ValueCls(ValueClsType.INT, unused, instr.tok), instr.arg1, tok=instr.tok))

            case InstructionType.PUSH_BOOL:
                stack.append(StackEntry(ValueCls(ValueClsType.BOOL, unused, instr.tok), instr.arg1, tok=instr.tok))

            case InstructionType.WORD:
                proc = ir.get_proc_by_name(instr.arg1)
                if proc is None:
                    typecheck_has_a_bug(instr, f'unknown word: {instr.arg1}')
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

            case InstructionType.INTRINSIC:
                intr_type = cast(IntrinsicType, instr.arg1)
                expect_enum_size(IntrinsicType, 27)
                match intr_type:
                    # arithmetic:
                    case IntrinsicType.ADD:
                        # a b -- a+b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 2)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(StackEntry(a.type, a.val + b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.SUB:
                        # a b -- a-b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 2)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(StackEntry(a.type, a.val - b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.MUL:
                        # a b -- a*b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 2)
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
                        expect_enum_size(ValueClsType, 2)
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
                        expect_enum_size(ValueClsType, 2)
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
                        expect_enum_size(ValueClsType, 2)
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
                        expect_enum_size(ValueClsType, 2)
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
                        expect_enum_size(ValueClsType, 2)
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
                        expect_enum_size(ValueClsType, 2)
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
                        expect_enum_size(ValueClsType, 2)
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
                        expect_enum_size(ValueClsType, 2)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(StackEntry(a.type, a.val ^ b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.BNOT:
                        # a -- ~a
                        if len(stack) < 1:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 2)
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
                        expect_enum_size(ValueClsType, 2)
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
                        expect_enum_size(ValueClsType, 2)
                        if a.type.type == b.type.type == ValueClsType.BOOL:
                            stack.append(StackEntry(a.type, a.val or b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected two BOOLs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.NOT:
                        # a -- (not a)
                        if len(stack) < 1:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 2)
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
                        expect_enum_size(ValueClsType, 2)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(
                                StackEntry(
                                    ValueCls(ValueClsType.BOOL, unused, instr.tok), a.val == b.val, tok=instr.tok
                                )
                            )

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.NE:
                        # a b -- (a != b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 2)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(
                                StackEntry(
                                    ValueCls(ValueClsType.BOOL, unused, instr.tok), a.val != b.val, tok=instr.tok
                                )
                            )

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.LT:
                        # a b -- (a < b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 2)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(
                                StackEntry(ValueCls(ValueClsType.BOOL, unused, instr.tok), a.val < b.val, tok=instr.tok)
                            )

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.GT:
                        # a b -- (a > b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 2)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(
                                StackEntry(ValueCls(ValueClsType.BOOL, unused, instr.tok), a.val > b.val, tok=instr.tok)
                            )

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.LE:
                        # a b -- (a <= b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 2)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(
                                StackEntry(
                                    ValueCls(ValueClsType.BOOL, unused, instr.tok), a.val <= b.val, tok=instr.tok
                                )
                            )

                        else:
                            typecheck_has_a_bug(instr, f'expected two INTs, got {pp(a.type)} and {pp(b.type)}')

                    case IntrinsicType.GE:
                        # a b -- (a >= b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueClsType, 2)
                        if a.type.type == b.type.type == ValueClsType.INT:
                            stack.append(
                                StackEntry(
                                    ValueCls(ValueClsType.BOOL, unused, instr.tok), a.val >= b.val, tok=instr.tok
                                )
                            )

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

                    # debugging:
                    case IntrinsicType.PRINT:
                        # a --
                        if len(stack) < 1:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        a = stack.pop()
                        print(f'[PRINT] {pp(a)}')

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
            case _:
                assert_never(type.type)

    def declare_var(var: str, type: ValueCls) -> None:
        typ: str = c_type(type)
        sb_header.add(f'  {typ} {var};\n')

    def copy_stacks(src: Stack, dst: Stack) -> None:
        for a, b in zip(src, dst):
            if a.type != b.type:
                unreachable(instr)

            if a.val == b.val:
                continue

            expect_enum_size(ValueClsType, 2)
            if a.type.type == b.type.type == ValueClsType.INT:
                sb.add(f'{'':{indent}}{b.val} = {a.val};\n')

            elif a.type.type == b.type.type == ValueClsType.BOOL:
                sb.add(f'{'':{indent}}{b.val} = {a.val};\n')

            else:
                unreachable(instr)

    block_stack: list[Block] = []
    sb_global = StringBuilder()
    sb_global += f'#include <stdio.h>\n'

    for proc in ir.procs:
        sb = StringBuilder()
        sb_header = StringBuilder()
        sb_struct = StringBuilder()

        instrs = proc.instrs
        stack: Stack = []

        ret = f'ret_{proc.name}'
        sb_header += f'{ret} {proc.name}('
        for i, t in enumerate(proc.ins):
            name = get_varname('arg')
            if i > 0:
                sb_header += ', '
            sb_header += f'{c_type(t)} {name}'
            stack.append(StackEntry(t, name, t.tok))
        sb_header += ') {\n'
        sb_struct += f'struct {ret} {{\n'
        for i, t in enumerate(proc.outs):
            name = f'_{i}'
            sb_struct += f'  {c_type(t)} {name};\n'
        sb_struct += '};\n'
        indent = 2

        for ip, instr in enumerate(instrs):
            ip += 1
            match instr.type:
                case InstructionType.PUSH_INT:
                    var = get_varname('lit_int')
                    typ = ValueCls(ValueClsType.INT, unused, instr.tok)
                    declare_var(var, typ)
                    stack.append(StackEntry(typ, var, tok=instr.tok))
                    sb += f'{'':{indent}}{var} = {instr.tok.value};\n'

                case InstructionType.PUSH_BOOL:
                    var = get_varname('lit_bool')
                    typ = ValueCls(ValueClsType.BOOL, unused, instr.tok)
                    declare_var(var, typ)
                    stack.append(StackEntry(typ, var, tok=instr.tok))
                    sb += f'{'':{indent}}{var} = {instr.tok.value};\n'

                case InstructionType.WORD:
                    proc_called = ir.get_proc_by_name(instr.arg1)
                    if proc_called is None:
                        unreachable(instr, f'proc {instr.arg1} not found')

                    ret_var = f'res_{proc_called.name}'
                    ret_type = f'ret_{proc_called.name}'
                    sb += f'{'':{indent}}{ret_type} {ret_var} = {proc_called.name}('
                    for i, arg in enumerate(stack[len(stack) - len(proc_called.ins) :]):
                        if i > 0:
                            sb += ', '
                        sb += f'{arg.val}'
                        _ = stack.pop()
                    sb += ');\n'

                    for i, out in enumerate(proc_called.outs):
                        typ = ValueCls(out.type, unused, instr.tok)
                        var = get_varname(f'res_{proc_called.name}_{i}')
                        declare_var(var, typ)
                        stack.append(StackEntry(typ, var, tok=instr.tok))
                        sb += f'{'':{indent}}{var} = {ret_var}._{i};\n'
                    # stack.append(StackEntry(typ, var, tok=instr.tok))
                    # sb += f'{'':{indent}}{var} = {proc_called.name};\n'

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
                    ret_type = f'ret_{proc.name}'
                    sb += f'{'':{indent}}return {ret_type} {{\n'
                    indent += 2
                    for i, x in enumerate(stack):
                        sb += f'{'':{indent}}._{i} = {x.val},\n'
                    indent -= 2
                    sb += f'{'':{indent}}}};\n'
                    stack = []

                case InstructionType.LABEL:
                    notimplemented(instr, f'translating {InstructionType.LABEL}')

                case InstructionType.INTRINSIC:
                    intr_type = cast(IntrinsicType, instr.arg1)
                    expect_enum_size(IntrinsicType, 27)
                    match intr_type:
                        # arithmetic:
                        case IntrinsicType.ADD:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 2)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'add')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} + {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.SUB:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 2)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'sub')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} - {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.MUL:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 2)
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
                            expect_enum_size(ValueClsType, 2)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'div')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} / {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.MOD:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 2)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'mod')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} % {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.DIVMOD:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 2)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var1 = get_varname(f'div')
                                var2 = get_varname(f'mod')
                                declare_var(var1, a.type)
                                declare_var(var2, a.type)
                                stack.append(StackEntry(a.type, var1, tok=instr.tok))
                                stack.append(StackEntry(a.type, var2, tok=instr.tok))
                                sb += f'{'':{indent}}{var1} = {a.val} / {b.val};\n'
                                sb += f'{'':{indent}}{var2} = {a.val} % {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.SHL:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 2)
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
                            expect_enum_size(ValueClsType, 2)
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
                            expect_enum_size(ValueClsType, 2)
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
                            expect_enum_size(ValueClsType, 2)
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
                            expect_enum_size(ValueClsType, 2)
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
                            expect_enum_size(ValueClsType, 2)
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
                            expect_enum_size(ValueClsType, 2)
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
                            expect_enum_size(ValueClsType, 2)
                            if a.type.type == b.type.type == ValueClsType.BOOL:
                                var = get_varname(f'or')
                                declare_var(var, a.type)
                                stack.append(StackEntry(a.type, var, tok=instr.tok))
                                sb += f'{'':{indent}}{var} = {a.val} || {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.NOT:
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 2)
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
                            expect_enum_size(ValueClsType, 2)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'eq')
                                declare_var(var, a.type)
                                stack.append(
                                    StackEntry(ValueCls(ValueClsType.BOOL, unused, instr.tok), var, tok=instr.tok)
                                )
                                sb += f'{'':{indent}}{var} = {a.val} == {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.NE:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 2)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'ne')
                                declare_var(var, a.type)
                                stack.append(
                                    StackEntry(ValueCls(ValueClsType.BOOL, unused, instr.tok), var, tok=instr.tok)
                                )
                                sb += f'{'':{indent}}{var} = {a.val} != {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.LT:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 2)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'lt')
                                declare_var(var, a.type)
                                stack.append(
                                    StackEntry(ValueCls(ValueClsType.BOOL, unused, instr.tok), var, tok=instr.tok)
                                )
                                sb += f'{'':{indent}}{var} = {a.val} < {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.GT:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 2)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'gt')
                                declare_var(var, a.type)
                                stack.append(
                                    StackEntry(ValueCls(ValueClsType.BOOL, unused, instr.tok), var, tok=instr.tok)
                                )
                                sb += f'{'':{indent}}{var} = {a.val} > {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.LE:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 2)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'le')
                                declare_var(var, a.type)
                                stack.append(
                                    StackEntry(ValueCls(ValueClsType.BOOL, unused, instr.tok), var, tok=instr.tok)
                                )
                                sb += f'{'':{indent}}{var} = {a.val} <= {b.val};\n'

                            else:
                                unreachable(instr)

                        case IntrinsicType.GE:
                            b = stack.pop()
                            a = stack.pop()
                            expect_enum_size(ValueClsType, 2)
                            if a.type.type == b.type.type == ValueClsType.INT:
                                var = get_varname(f'ge')
                                declare_var(var, a.type)
                                stack.append(
                                    StackEntry(ValueCls(ValueClsType.BOOL, unused, instr.tok), var, tok=instr.tok)
                                )
                                sb += f'{'':{indent}}{var} = {a.val} >= {b.val};\n'

                            else:
                                unreachable(instr)

                        # stack manipulation:
                        case IntrinsicType.DUP:
                            # a -- a a
                            a = stack.pop()
                            stack.append(a)

                            var = get_varname(f'dup')
                            declare_var(var, a.type)
                            stack.append(StackEntry(a.type, var, tok=instr.tok))
                            sb += f'{'':{indent}}{var} = {a.val};\n'

                        case IntrinsicType.DROP:
                            # a --
                            _ = stack.pop()

                        case IntrinsicType.SWAP:
                            # a b -- b a
                            b = stack.pop()
                            a = stack.pop()
                            stack.append(b)
                            stack.append(a)

                        case IntrinsicType.ROT:
                            # a b c -- b c a
                            c = stack.pop()
                            b = stack.pop()
                            a = stack.pop()
                            stack.append(b)
                            stack.append(c)
                            stack.append(a)

                        # debugging:
                        case IntrinsicType.PRINT:
                            # a --
                            a = stack.pop()
                            sb += f'{'':{indent}}printf("%d\\n", {a.val});\n'

                        case IntrinsicType.DEBUG:
                            pass

                        case _:
                            # sb += f'  // {pp(instr)}'
                            assert_never(intr_type)
                case _:
                    assert_never(instr.type)

        if stack:
            unreachable(None, stack=stack)

        sb += '}\n'
        sb_global += str(sb_struct)
        sb_global += str(sb_header)
        sb_global += str(sb)

    if block_stack:
        unreachable(None, block_stack=block_stack)

    code = str(sb_global)
    return code


def repl() -> None:
    import traceback

    while True:
        try:
            line = input('> ')
            if line == 'q':
                break

            loc = Loc('<repl>', 0, 0)
            toks = [*tokenize(line, filename='<repl>')]
            ir = compile(toks)
            trace(
                loc,
                'before typechecking and running',
                Tokens=toks,
                IR=ir,
            )
            typecheck(ir)
            interpret(ir)

        except EOFError:
            break
        except SystemExit:
            pass
        except Exception:
            traceback.print_exc()


def run_file(file: str) -> None:
    with open(file, 'rt') as f:
        text = f.read()

    loc = Loc(file, 0, 0)
    toks = [*tokenize(text, filename=file)]
    trace(
        loc,
        f'file: {file}',
        Tokens=toks,
    )
    ir = compile(toks)
    trace(
        loc,
        f'file: {file}',
        IR=ir,
    )
    typecheck(ir)
    gen_code = translate(ir)

    trace(
        loc,
        'Generated C code:',
        generated_code=gen_code,
    )
    try:
        interpret(ir)
    except Exception:
        traceback.print_exc()
        raise SystemExit(69)


def main(argv: list[str]) -> None:
    def usage_short() -> None:
        print(f'Usage: py lang.py [OPTIONS] SUBCOMMAND <ARGS>')

    def usage() -> None:
        usage_short()
        print(f'Options:')
        print(f'  -h --help       print this help message')
        print(f'  -l <level>      log level: ERROR,WARN,INFO,TRACE')
        print(f'Subcommands:')
        print(f'  run FILE        run a file')
        print(f'    FILE            a file to run')
        print(f'  repl            start a Read-Eval-Print-Loop')

    global log_level

    while argv:
        if argv[0] == '-h' or argv[0] == '--help':
            usage()
            sys.exit(0)

        elif argv[0] == '-l':
            _, *argv = argv
            if len(argv) < 1:
                usage_short()
                print(f'[ERROR] no log level specified')
                sys.exit(2)

            ll_str, *argv = argv
            if ll_str not in LL:
                error(Loc('<cli>', 1, 0), f'invalid log level: {ll_str}, expected one of {list(LL)}')
            log_level = LL[ll_str]

        else:
            break

    if len(argv) < 1:
        usage_short()
        print(f'[ERROR] no subcommand specified')
        sys.exit(2)

    subcmd, *argv = argv

    if subcmd == 'run':
        while len(argv) > 0:
            if argv[0] == '-h':
                usage()
                sys.exit(0)

            else:
                break

        if len(argv) < 1:
            usage_short()
            print(f'[ERROR] no file specified')
            sys.exit(2)

        filename, *argv = argv

        if len(argv) > 0:
            usage_short()
            print(f'[ERROR] unrecognized arguments: {argv}')
            sys.exit(2)
        run_file(filename)
        sys.exit(0)

    elif subcmd == 'repl':
        repl()
        sys.exit(0)

    else:
        usage_short()
        print(f'[ERROR] unknown subcommand: {subcmd}')
        sys.exit(2)


if __name__ == '__main__':
    main(sys.argv[1:])
