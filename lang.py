from __future__ import annotations

from collections.abc import Iterable
import enum
from typing import Any, Never, assert_never, cast, override
from enum import Enum, auto
from dataclasses import dataclass


class _sentinel:
    def __init__(self, name: str) -> None:
        self.name = name

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
    def __str__(self) -> str:
        return f'{self.file}:{self.line}:{self.col}'


class TokenType(Enum):
    INT = auto()
    BOOL = auto()
    CHAR = auto()
    STR = auto()
    # FLOAT = auto()
    WORD = auto()
    KEYWORD = auto()


class KeywordType(Enum):
    PROC = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    DO = auto()
    END = auto()


@dataclass
class Token:
    type: TokenType
    value: Any
    loc: Loc


class InstructionType(Enum):
    PUSH_INT = auto()
    PUSH_BOOL = auto()
    WORD = auto()
    INTRINSIC = auto()

    PROC = auto()
    IF = auto()
    ELSE = auto()
    WHILE = auto()
    DO = auto()
    END = auto()


class IntrinsicType(Enum):
    # arithhmetic:
    ADD = auto()  # a b -- a+b
    SUB = auto()  # a b -- a-b
    MUL = auto()  # a b -- a*b
    DIV = auto()  # a b -- a//b
    MOD = auto()  # a b -- a%b
    DIVMOD = auto()  # a b -- a//b a%b

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
    tok: Token
    arg1: Any = unused
    arg2: Any = unused


class ValueType(Enum):
    INT = auto()
    BOOL = auto()


@dataclass
class StackEntry:
    type: ValueType
    val: Any
    tok: Token


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
                return f'{pp(x.type)}'
            return f'{pp(x.type)}:{pp(x.val)}'

        case Token():
            return f'{x.loc}:{pp(x.type)}:{pp(x.value)}'

        case Instruction():
            s = f'{x.type.name}'
            if x.arg1 is not unused:
                s += f':{pp(x.arg1)}'
            if x.arg2 is not unused:
                s += f':{pp(x.arg2)}'
            return s

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
            return Loc('<?>', 0, 0)
        case _:
            assert_never(loc)


def note(**notes: Any) -> None:
    for k, v in notes.items():
        print(f'[NOTE] {k}:')
        match v:
            case list():
                if not v:
                    print(f'         (empty)')
                for i, x in enumerate(v):
                    print(f'         {i:3}. {pp(x)}')

            case str():
                for line in v.splitlines():
                    print(f'         {line}')

            case _:
                for line in pp(v).splitlines():
                    print(f'         {line}')


def error(loc: _Locatable, msg: str, **notes) -> Never:
    print(f'[ERROR] {_locatable_to_loc(loc)}: {msg}')
    note(**notes)
    raise Error()


def warn(loc: _Locatable, msg: str, **notes) -> None:
    print(f'[WARNING] {_locatable_to_loc(loc)}: {msg}')
    note(**notes)


def info(loc: _Locatable, msg: str, **notes) -> None:
    print(f'[INFO] {_locatable_to_loc(loc)}: {msg}')
    note(**notes)


def typecheck_has_a_bug(loc: _Locatable, msg: str, **notes) -> Never:
    error(
        loc,
        f'this should not happen because this must have been caugth at typecheck time: {msg}',
        **notes,
    )


def unreachable(loc: _Locatable, msg: str, **notes) -> Never:
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
            elif chunk == 'true' or chunk == 'false':
                yield Token(TokenType.BOOL, chunk == 'true', loc_start)
            elif is_int(chunk):
                yield Token(TokenType.INT, int(chunk), loc_start)
            else:
                match chunk:
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
                    case _:
                        yield Token(TokenType.WORD, chunk, loc_start)

            loc_start = Loc(filename, line_no, col_no)
            i_start = i + 1

        i += 1


def compile(toks: list[Token]) -> list[Instruction]:
    result: list[Instruction] = []

    @dataclass
    class Block:
        type: InstructionType
        ip1: int = -1
        ip2: int = -1
        ip3: int = -1
        ip4: int = -1

    block_stack: list[Block] = []

    for tok in toks:
        expect_enum_size(TokenType, 6)
        match tok.type:
            case TokenType.INT:
                result.append(Instruction(type=InstructionType.PUSH_INT, tok=tok, arg1=tok.value))

            case TokenType.BOOL:
                result.append(Instruction(type=InstructionType.PUSH_BOOL, tok=tok, arg1=tok.value))

            case TokenType.CHAR:
                if len(tok.value) != 1:
                    error(
                        tok,
                        f'invalid character literal: {tok.value}, it must contain exactly one character',
                    )
                result.append(
                    Instruction(type=InstructionType.PUSH_INT, tok=tok, arg1=ord(tok.value))
                )

            case TokenType.STR:
                notimplemented(tok.loc, f'string literals')

            case TokenType.KEYWORD:
                kw_type = cast(KeywordType, tok.value)
                expect_enum_size(KeywordType, 6)
                match kw_type:
                    case KeywordType.IF:
                        b = Block(
                            InstructionType.IF,
                        )
                        block_stack.append(b)
                        b.ip1 = len(result)

                        result.append(Instruction(type=InstructionType.IF, tok=tok))

                    case KeywordType.ELSE:
                        if not block_stack:
                            error(tok, f'ELSE should follow an IF')
                        b = block_stack[-1]
                        if b.type != InstructionType.IF:
                            error(tok, f'ELSE should follow an IF, not {b.type}')
                        b.ip3 = len(result)

                        result.append(Instruction(type=InstructionType.ELSE, tok=tok, arg1=missing))

                    case KeywordType.WHILE:
                        b = Block(
                            InstructionType.WHILE,
                        )
                        block_stack.append(b)
                        b.ip1 = len(result)

                        result.append(Instruction(type=InstructionType.WHILE, tok=tok))

                    case KeywordType.DO:
                        if not block_stack:
                            error(tok, f'DO should follow an IF or WHILE')
                        b = block_stack[-1]
                        if b.type == InstructionType.WHILE:
                            b.ip2 = len(result)
                            result.append(
                                Instruction(type=InstructionType.DO, tok=tok, arg1=missing)
                            )

                        elif b.type == InstructionType.IF:
                            b.ip2 = len(result)
                            result.append(
                                Instruction(type=InstructionType.DO, tok=tok, arg1=missing)
                            )

                        else:
                            error(tok, f'DO should follow an IF or WHILE, not {b.type}')

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
                                    b.ip3 = len(result)
                                    result.append(
                                        Instruction(
                                            type=InstructionType.ELSE, tok=tok, arg1=missing
                                        )
                                    )

                                b.ip4 = len(result)
                                result[b.ip2].arg1 = b.ip3
                                result[b.ip3].arg1 = b.ip4
                                result.append(
                                    Instruction(type=InstructionType.END, tok=tok, arg1=b.ip4 + 1)
                                )

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

                                b.ip3 = len(result)

                                result[b.ip2].arg1 = b.ip3
                                result.append(
                                    Instruction(type=InstructionType.END, tok=tok, arg1=b.ip1)
                                )

                            case _:
                                error(tok, f'END should follow an IF or WHILE, not {b.type}')

                    case KeywordType.PROC:
                        notimplemented(tok, '`proc` keywords')

                    case _:
                        assert_never(kw_type)

            case TokenType.WORD:
                expect_enum_size(IntrinsicType, 27)
                match tok.value:
                    # arithmetic:
                    case '+':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.ADD
                            )
                        )
                    case '-':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.SUB
                            )
                        )
                    case '*':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.MUL
                            )
                        )
                    case '/':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.DIV
                            )
                        )
                    case '%':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.MOD
                            )
                        )
                    case 'divmod':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.DIVMOD
                            )
                        )

                    case '<<':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.SHL
                            )
                        )
                    case '>>':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.SHR
                            )
                        )

                    # bitwise:
                    case '&':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.BAND
                            )
                        )
                    case '|':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.BOR
                            )
                        )
                    case '^':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.BXOR
                            )
                        )
                    case '~':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.BNOT
                            )
                        )

                    # logic:
                    case 'and':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.AND
                            )
                        )
                    case 'or':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.OR
                            )
                        )
                    case 'not':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.NOT
                            )
                        )

                    # comparison:
                    case '==':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.EQ
                            )
                        )
                    case '!=':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.NE
                            )
                        )
                    case '<':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.LT
                            )
                        )
                    case '>':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.GT
                            )
                        )
                    case '<=':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.LE
                            )
                        )
                    case '>=':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.GE
                            )
                        )

                    # stack manipulation:
                    case 'dup':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.DUP
                            )
                        )
                    case 'drop':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.DROP
                            )
                        )
                    case 'swap':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.SWAP
                            )
                        )
                    case 'rot':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.ROT
                            )
                        )

                    # debugging:
                    case 'print':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.PRINT
                            )
                        )
                    case '?':
                        result.append(
                            Instruction(
                                type=InstructionType.INTRINSIC, tok=tok, arg1=IntrinsicType.DEBUG
                            )
                        )

                    case _:
                        result.append(
                            Instruction(type=InstructionType.WORD, tok=tok, arg1=tok.value)
                        )

            case _:
                assert_never(tok.type)

    if block_stack:
        error(
            None,
            'unclosed blocks',
            blocks=block_stack,
        )

    return result


def typecheck(instrs: list[Instruction]) -> None:
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

    block_stack: list[Block] = []

    stack: Stack = []
    ip = 0
    while ip < len(instrs):
        instr = instrs[ip]
        ip += 1
        expect_enum_size(InstructionType, 10)
        match instr.type:
            case InstructionType.PUSH_INT:
                stack.append(StackEntry(ValueType.INT, typechecking, tok=instr.tok))

            case InstructionType.PUSH_BOOL:
                stack.append(StackEntry(ValueType.BOOL, typechecking, tok=instr.tok))

            case InstructionType.WORD:
                error(
                    instr,
                    f'arbitrary word handling is not implemented yet: {instr.tok.value}',
                )

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
                        unreachable(
                            instr, f'{InstructionType.IF} block doesnt have stack1 saved up'
                        )
                    stack = block.stack1

                else:
                    error(instr, f'{InstructionType.ELSE} must come after {InstructionType.IF}')

            case InstructionType.END:
                block = block_stack.pop()
                if block.type == InstructionType.IF:
                    if block.stack2 is None:
                        unreachable(
                            instr, f'{InstructionType.IF} block doesnt have stack2 saved up'
                        )
                    if not compare_stacks(block.stack2, stack):
                        error(
                            instr,
                            f'both branches of {InstructionType.IF} must leave the stack in the same state',
                            stack_then=block.stack2,
                            stack_else=stack,
                        )

                elif block.type == InstructionType.WHILE:
                    if block.stack1 is None:
                        unreachable(
                            instr, f'{InstructionType.WHILE} block doesnt have stack1 saved up'
                        )
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
                if a.type == ValueType.BOOL:
                    block = block_stack[-1]
                    if block.type == InstructionType.IF:
                        if block.stack1 is None:
                            unreachable(
                                instr, f'{InstructionType.IF} block doesnt have stack1 saved up'
                            )
                        if not compare_stacks(block.stack1, stack):
                            error(
                                instr,
                                f'condition part of {InstructionType.IF} should put exactly one BOOL on the stack',
                                expected_stack=block.stack1,
                                actual_stack=stack,
                            )

                    elif block.type == InstructionType.WHILE:
                        if block.stack1 is None:
                            unreachable(
                                instr, f'{InstructionType.WHILE} block doesnt have stack1 saved up'
                            )
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
                        f'{InstructionType.DO} expects one BOOL on the stack, but got {a.type}',
                        other_stack_items=stack,
                    )

            case InstructionType.PROC:
                notimplemented(instr, f'typechecking {InstructionType.PROC}')

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
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.INT, typechecking, tok=instr.tok))

                        else:
                            error(
                                instr,
                                f'cannot add/subtract {a.type} and {b.type}',
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
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.INT, typechecking, tok=instr.tok))

                        else:
                            error(
                                instr,
                                f'cannot multiply/divide {a.type} by {b.type}',
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
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.INT, typechecking, tok=instr.tok))

                        else:
                            error(
                                instr,
                                f'cannot modulo {a.type} by {b.type}',
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
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.INT, typechecking, tok=instr.tok))
                            stack.append(StackEntry(ValueType.INT, typechecking, tok=instr.tok))

                        else:
                            error(
                                instr,
                                f'cannot divmod {a.type} by {b.type}',
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
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.INT, typechecking, tok=instr.tok))

                        else:
                            error(
                                instr,
                                f'cannot shift {a.type} by {b.type}',
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
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.INT, typechecking, tok=instr.tok))

                        else:
                            error(
                                instr,
                                f'cannot bitwise or/and/xor {a.type} with {b.type}',
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
                        if a.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.INT, typechecking, tok=instr.tok))

                        else:
                            error(
                                instr,
                                f'cannot bitwise not an {a.type}',
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
                        if a.type == b.type == ValueType.BOOL:
                            stack.append(StackEntry(ValueType.BOOL, typechecking, tok=instr.tok))

                        else:
                            error(
                                instr,
                                f'cannot and/or {a.type} with {b.type}',
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
                        if a.type == ValueType.BOOL:
                            stack.append(StackEntry(ValueType.BOOL, typechecking, tok=instr.tok))

                        else:
                            error(
                                instr,
                                f'cannot not {a.type}',
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
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.BOOL, typechecking, tok=instr.tok))

                        else:
                            error(
                                instr,
                                f'cannot compare {a.type} and {b.type}',
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
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.BOOL, typechecking, tok=instr.tok))

                        else:
                            error(
                                instr,
                                f'cannot compare {a.type} and {b.type}',
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


def interpret(instrs: Iterable[Instruction]) -> None:
    instrs = list(instrs)

    stack: list[StackEntry] = []
    ip = 0
    while ip < len(instrs):
        instr = instrs[ip]
        ip += 1
        expect_enum_size(InstructionType, 10)
        match instr.type:
            case InstructionType.PUSH_INT:
                stack.append(StackEntry(ValueType.INT, instr.arg1, tok=instr.tok))

            case InstructionType.PUSH_BOOL:
                stack.append(StackEntry(ValueType.BOOL, instr.arg1, tok=instr.tok))

            case InstructionType.WORD:
                notimplemented(
                    instr, f'arbitrary word handling is not implemented yet: {instr.arg1}'
                )

            case InstructionType.IF:
                pass

            case InstructionType.ELSE:
                ip = instr.arg1

            case InstructionType.END:
                ip = instr.arg1

            case InstructionType.WHILE:
                pass

            case InstructionType.DO:
                if len(stack) < 1:
                    typecheck_has_a_bug(instr, 'not enough items on stack')
                a = stack.pop()
                if a.type == ValueType.BOOL:
                    if a.val:
                        ip = ip
                    else:
                        ip = instr.arg1 + 1

                else:
                    typecheck_has_a_bug(instr, f'expected BOOL, got {a.type}')

            case InstructionType.PROC:
                notimplemented(instr, 'procedures are not implemented yet')

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
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.INT, a.val + b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected two INTs, got {a.type} and {b.type}'
                            )

                    case IntrinsicType.SUB:
                        # a b -- a-b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.INT, a.val - b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected two INTs, got {a.type} and {b.type}'
                            )

                    case IntrinsicType.MUL:
                        # a b -- a*b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.INT, a.val * b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected two INTs, got {a.type} and {b.type}'
                            )

                    case IntrinsicType.DIV:
                        # a b -- a/b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.INT, a.val // b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected two INTs, got {a.type} and {b.type}'
                            )

                    case IntrinsicType.MOD:
                        # a b -- a%b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.INT, a.val % b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected two INTs, got {a.type} and {b.type}'
                            )

                    case IntrinsicType.DIVMOD:
                        # a b -- a//b a%b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.INT, a.val // b.val, tok=instr.tok))
                            stack.append(StackEntry(ValueType.INT, a.val % b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected two INTs, got {a.type} and {b.type}'
                            )

                    case IntrinsicType.SHL:
                        # a b -- a<<b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.INT, a.val << b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected two INTs, got {a.type} and {b.type}'
                            )

                    case IntrinsicType.SHR:
                        # a b -- a>>b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.INT, a.val >> b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected two INTs, got {a.type} and {b.type}'
                            )

                    case IntrinsicType.BOR:
                        # a b -- a|b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.INT, a.val | b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected two INTs, got {a.type} and {b.type}'
                            )

                    case IntrinsicType.BAND:
                        # a b -- a&b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.INT, a.val & b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected two INTs, got {a.type} and {b.type}'
                            )

                    case IntrinsicType.BXOR:
                        # a b -- a^b
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.INT, a.val ^ b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected two INTs, got {a.type} and {b.type}'
                            )

                    case IntrinsicType.BNOT:
                        # a -- ~a
                        if len(stack) < 1:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.INT, ~a.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected an INT, got {a.type}')

                    # logic:
                    case IntrinsicType.AND:
                        # a b -- (a and b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.BOOL:
                            stack.append(StackEntry(ValueType.BOOL, a.val and b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected two BOOLs, got {a.type} and {b.type}'
                            )

                    case IntrinsicType.OR:
                        # a b -- (a or b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.BOOL:
                            stack.append(StackEntry(ValueType.BOOL, a.val or b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected two BOOLs, got {a.type} and {b.type}'
                            )

                    case IntrinsicType.NOT:
                        # a -- (not a)
                        if len(stack) < 1:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == ValueType.BOOL:
                            stack.append(StackEntry(ValueType.BOOL, not a.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(instr, f'expected a BOOL, got {a.type}')

                    # comparison:
                    case IntrinsicType.EQ:
                        # a b -- (a == b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.BOOL, a.val == b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected two INTs, got {a.type} and {b.type}'
                            )

                    case IntrinsicType.NE:
                        # a b -- (a != b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.BOOL, a.val != b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected two INTs, got {a.type} and {b.type}'
                            )

                    case IntrinsicType.LT:
                        # a b -- (a < b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.BOOL, a.val < b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected two INTs, got {a.type} and {b.type}'
                            )

                    case IntrinsicType.GT:
                        # a b -- (a > b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.BOOL, a.val > b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected two INTs, got {a.type} and {b.type}'
                            )

                    case IntrinsicType.LE:
                        # a b -- (a <= b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.BOOL, a.val <= b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected two INTs, got {a.type} and {b.type}'
                            )

                    case IntrinsicType.GE:
                        # a b -- (a >= b)
                        if len(stack) < 2:
                            typecheck_has_a_bug(instr, 'not enough items on stack')
                        b = stack.pop()
                        a = stack.pop()
                        expect_enum_size(ValueType, 2)
                        if a.type == b.type == ValueType.INT:
                            stack.append(StackEntry(ValueType.BOOL, a.val >= b.val, tok=instr.tok))

                        else:
                            typecheck_has_a_bug(
                                instr, f'expected two INTs, got {a.type} and {b.type}'
                            )

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
                        stack.pop()

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


def repl() -> None:
    import traceback
    import readline

    while True:
        try:
            line = input('> ')
            if line == 'q':
                break
            line += '\n'
            toks = [*tokenize(line, filename='<repl>')]
            ir = [*compile(toks)]
            note(
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
    toks = [*tokenize(text, filename=file)]
    ir = [*compile(toks)]
    typecheck(ir)
    interpret(ir)


def main() -> None:
    import sys

    args = sys.argv[1:]

    def unshift() -> str:
        res, args[:] = args[0], args[1:]
        return res

    def usage() -> None:
        print(f'Usage: {sys.executable} {sys.argv[0]} SUBCOMMAND [OPTIONS]')
        print(f'Subcommands:')
        print(f'  help        print this help message')
        print(f'  run FILE    run a file')
        print(f'  repl        start a Read-Eval-Print-Loop')

    file = ''

    if not args:
        usage()
        print(f'Error: no subcommand specified')
        sys.exit(1)

    arg = unshift()

    match arg:
        case 'help' | '-h' | '--help':
            usage()
            sys.exit(0)

        case 'run':
            if not args:
                usage()
                print(f'Error: no file specified')
                sys.exit(2)

            file = unshift()
            if args:
                usage()
                print(f'Error: unexpected arguments: {args}')
                sys.exit(2)

            run_file(file)
            sys.exit(0)

        case 'repl':
            if args:
                usage()
                print(f'Error: unexpected arguments: {args}')
                sys.exit(2)

            repl()
            sys.exit(0)

        case _:
            usage()
            sys.exit(2)

    assert_never(0)


if __name__ == '__main__':
    main()
