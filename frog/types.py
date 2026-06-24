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
        return f'({self.name})'


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
    LET = auto()


KW_TO_KWT = {
    'proc': KeywordType.PROC,
    'if': KeywordType.IF,
    'else': KeywordType.ELSE,
    'while': KeywordType.WHILE,
    'do': KeywordType.DO,
    'end': KeywordType.END,
    '--': KeywordType.TYPE_DELIM,
    'let': KeywordType.LET,
}
assert len(KW_TO_KWT) == len(KeywordType)


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

    BIND = auto()
    LOAD_BIND = auto()
    UNBIND = auto()


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
    DUP2 = auto()  # a b -- a b a b
    DROP = auto()  # a --
    SWAP = auto()  # a b -- b a
    SWAP2 = auto()  # a b x y -- x y a b
    ROT = auto()  # a b c -- b c a
    CAST = auto()  # x T -- (T)x

    # memory:
    ALLOC = auto()  # size -- ptr
    READ_I8 = auto()  # ptr -- int
    READ_I16 = auto()  # ptr -- int
    READ_I32 = auto()  # ptr -- int
    READ_I64 = auto()  # ptr -- int
    READ_U8 = auto()  # ptr -- int
    READ_U16 = auto()  # ptr -- int
    READ_U32 = auto()  # ptr -- int
    READ_U64 = auto()  # ptr -- int
    WRITE_I8 = auto()  # int ptr --
    WRITE_I16 = auto()  # int ptr --
    WRITE_I32 = auto()  # int ptr --
    WRITE_I64 = auto()  # int ptr --
    WRITE_U8 = auto()  # int ptr --
    WRITE_U16 = auto()  # int ptr --
    WRITE_U32 = auto()  # int ptr --
    WRITE_U64 = auto()  # int ptr --

    # debugging:
    PRINT = auto()  # a --
    PUTC = auto()  # int --
    DEBUG = auto()  # --


INTRINSIC_TO_INTRINSIC_TYPE = {
    '+': IntrinsicType.ADD,
    '-': IntrinsicType.SUB,
    '*': IntrinsicType.MUL,
    '/': IntrinsicType.DIV,
    '%': IntrinsicType.MOD,
    '/%': IntrinsicType.DIVMOD,
    '<<': IntrinsicType.SHL,
    '>>': IntrinsicType.SHR,
    '|': IntrinsicType.BOR,
    '&': IntrinsicType.BAND,
    '^': IntrinsicType.BXOR,
    '~': IntrinsicType.BNOT,
    '&&': IntrinsicType.AND,
    '||': IntrinsicType.OR,
    '!': IntrinsicType.NOT,
    '==': IntrinsicType.EQ,
    '!=': IntrinsicType.NE,
    '<': IntrinsicType.LT,
    '>': IntrinsicType.GT,
    '<=': IntrinsicType.LE,
    '>=': IntrinsicType.GE,
    'dup': IntrinsicType.DUP,
    'dup2': IntrinsicType.DUP2,
    'drop': IntrinsicType.DROP,
    'swap': IntrinsicType.SWAP,
    'swap2': IntrinsicType.SWAP2,
    'rot': IntrinsicType.ROT,
    'cast': IntrinsicType.CAST,
    'alloc': IntrinsicType.ALLOC,
    '@i8': IntrinsicType.READ_I8,
    '@i16': IntrinsicType.READ_I16,
    '@i32': IntrinsicType.READ_I32,
    '@i64': IntrinsicType.READ_I64,
    '@u8': IntrinsicType.READ_U8,
    '@u16': IntrinsicType.READ_U16,
    '@u32': IntrinsicType.READ_U32,
    '@u64': IntrinsicType.READ_U64,
    '!i8': IntrinsicType.WRITE_I8,
    '!i16': IntrinsicType.WRITE_I16,
    '!i32': IntrinsicType.WRITE_I32,
    '!i64': IntrinsicType.WRITE_I64,
    '!u8': IntrinsicType.WRITE_U8,
    '!u16': IntrinsicType.WRITE_U16,
    '!u32': IntrinsicType.WRITE_U32,
    '!u64': IntrinsicType.WRITE_U64,
    'print': IntrinsicType.PRINT,
    'putc': IntrinsicType.PUTC,
    '?': IntrinsicType.DEBUG,
}


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
