from typing import Any, final, override
from enum import Enum, auto
from dataclasses import dataclass, field, is_dataclass

from .sb import StringBuilder


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
    PUSH_TYPE = auto()

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
    CAST = auto()  # x T -- (T)x

    # debugging:
    PRINT = auto()  # a --
    DEBUG = auto()  # --


@dataclass
class Instruction:
    type: InstructionType
    tok: Token = field(compare=False)
    arg1: Any = unused
    arg2: Any = unused


class ValueClsType(Enum):
    INT = auto()
    BOOL = auto()
    PTR = auto()
    TYPE = auto()


@dataclass
class ValueCls:
    type: ValueClsType
    val: Any = unused


@dataclass
class Contract:
    ins: list[ValueCls]
    outs: list[ValueCls]


@dataclass
class Proc:
    name: str
    contract: Contract
    instrs: list[Instruction]
    tok: Token


@dataclass
class IR:
    procs: list[Proc] = field(default_factory=list)

    def get_proc_by_name(self, name: str) -> Proc | None:
        for proc in self.procs:
            if proc.name == name:
                return proc
        return None


@dataclass
class StackEntry:
    type: ValueCls
    val: Any
    tok: Token = field(compare=False)


class Error(Exception): ...


type Stack = list[StackEntry]


def pp(x: Any) -> str:
    match x:
        case Enum():
            return x.name

        case StackEntry():
            if x.val is typechecking:
                return f'{pp(x.type)}@{pp(x.tok.loc)}'
            return f'{pp(x.type)}:{pp(x.val)}@{pp(x.tok.loc)}'

        case ValueCls():
            if x.val is unused:
                return f'{pp(x.type)}()'
            return f'{pp(x.type)}({pp(x.val)})'

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
