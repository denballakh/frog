import json
from pathlib import Path
import re
from typing import cast

ROOT = Path(__file__).parent.parent
GRAMMAR = ROOT / 'ide' / 'vscode' / 'frog_grammar.json'


def dictionary(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def patterns(value: object) -> list[dict[str, object]]:
    assert isinstance(value, list)
    return [dictionary(item) for item in cast(list[object], value)]


def named_pattern(items: list[dict[str, object]], name: str) -> dict[str, object]:
    matches = [item for item in items if item.get('name') == name]
    assert len(matches) == 1
    return matches[0]


def main() -> None:
    grammar = dictionary(json.loads(GRAMMAR.read_text()))
    root_patterns = patterns(grammar['patterns'])
    assert sum(pattern.get('include') == '#literals' for pattern in root_patterns) == 1
    root_includes = [pattern.get('include') for pattern in root_patterns]
    assert root_includes.index('#special') < root_includes.index('#types')
    assert root_includes.index('#literals') < root_includes.index('#operators')

    repository = dictionary(grammar['repository'])
    literals = dictionary(repository['literals'])
    literal_patterns = patterns(literals['patterns'])
    character = named_pattern(literal_patterns, 'constant.character.frog')
    string = named_pattern(literal_patterns, 'string.quoted.double.frog')
    numeric = named_pattern(literal_patterns, 'constant.numeric.frog')
    operators = dictionary(repository['operators'])
    operator = named_pattern(patterns(operators['patterns']), 'keyword.operator.frog')
    keywords = dictionary(repository['keywords'])
    keyword = named_pattern(patterns(keywords['patterns']), 'keyword.control.frog')
    types_repository = dictionary(repository['types'])
    type_pattern = named_pattern(patterns(types_repository['patterns']), 'storage.type.frog')
    special = dictionary(repository['special'])
    special_function = named_pattern(patterns(special['patterns']), 'support.function')

    assert string['begin'] == '"'
    assert string['end'] == '"'
    assert 'match' not in string

    character_pattern = character['match']
    assert isinstance(character_pattern, str)
    assert '"|[^\'"\\r\\n]' in character_pattern
    character_regex = re.compile(character_pattern)
    assert character_regex.fullmatch("'E'")
    assert character_regex.fullmatch("'é'")
    assert character_regex.fullmatch("'\"'")
    assert character_regex.fullmatch(r"'\'")
    assert character_regex.fullmatch(r"'\''")
    assert character_regex.fullmatch("''") is None
    assert character_regex.fullmatch("'EE'") is None
    assert character_regex.fullmatch("'\\n'") is None

    operator_pattern = operator['match']
    assert isinstance(operator_pattern, str)
    assert re.compile(operator_pattern).fullmatch('!ptr')

    keyword_pattern = keyword['match']
    special_pattern = special_function['match']
    assert isinstance(keyword_pattern, str)
    assert isinstance(special_pattern, str)
    type_regex = re.compile(cast(str, type_pattern['match']))
    for spelling in ('String', 'int', 'i8', 'i16', 'i32', 'i64', 'u8', 'u16', 'u32', 'u64', 'bool', 'ptr'):
        assert type_regex.fullmatch(spelling), spelling
        assert type_regex.fullmatch(spelling + '*'), spelling + '*'
        assert type_regex.fullmatch(spelling + '**'), spelling + '**'
    assert re.compile(keyword_pattern).fullmatch('record')
    assert re.compile(keyword_pattern).fullmatch('union')
    assert re.compile(keyword_pattern).fullmatch('case')
    assert re.compile(keyword_pattern).fullmatch('fn')
    assert re.compile(keyword_pattern).fullmatch('const')
    assert re.compile(keyword_pattern).fullmatch('peek')
    assert re.compile(keyword_pattern).fullmatch('c-include')
    assert re.compile(keyword_pattern).fullmatch('c-type')
    assert re.compile(keyword_pattern).fullmatch('c-call')
    assert re.compile(keyword_pattern).fullmatch('c-value')
    assert type_regex.fullmatch('LegacyAbiType') is None
    special_regex = re.compile(special_pattern)
    assert special_regex.fullmatch('Node:alloc')
    assert special_regex.fullmatch('Node:sizeof')
    assert special_regex.fullmatch('@Node.bytes')
    assert special_regex.fullmatch('!Node.bytes')
    assert special_regex.fullmatch('@.bytes')
    assert special_regex.fullmatch('!.bytes')
    assert special_regex.fullmatch('Maybe:some')
    assert special_regex.fullmatch('Maybe.some?')
    assert special_regex.fullmatch('Mapper:ref:inc')
    assert special_regex.fullmatch('F:ref:plus-one')
    assert special_regex.fullmatch('F:ref:inc"quoted')
    assert special_regex.fullmatch('Mapper:call')
    assert special_regex.fullmatch('String.bytes')
    assert special_regex.fullmatch('String.len')
    assert special_regex.fullmatch('?')
    for library_word in ('alloc', 'putc', 'getc', 'eputc', 'exit', 'read-file'):
        assert special_regex.fullmatch(library_word) is None
    assert special_regex.fullmatch('Node.bytes') is None
    assert special_regex.fullmatch('Node.bytes!') is None
    assert special_regex.fullmatch('@.') is None
    assert special_regex.fullmatch('!.') is None
    assert special_regex.fullmatch('Maybe.some') is None

    string_patterns = patterns(string['patterns'])
    assert all('include' not in pattern for pattern in string_patterns)
    escape = named_pattern(string_patterns, 'constant.character.escape.frog')
    invalid_escape = named_pattern(string_patterns, 'invalid.illegal.escape.frog')
    assert string_patterns.index(escape) < string_patterns.index(invalid_escape)

    escape_pattern = escape['match']
    invalid_escape_pattern = invalid_escape['match']
    assert isinstance(escape_pattern, str)
    assert isinstance(invalid_escape_pattern, str)
    escape_regex = re.compile(escape_pattern)
    invalid_escape_regex = re.compile(invalid_escape_pattern)

    for spelling in (r'\\', r'\"', r'\n', r'\r', r'\t', r'\0', r'\x00', r'\xAf'):
        assert escape_regex.fullmatch(spelling), spelling
    for spelling in (r'\q', r'\x', r'\x0', r'\xGG'):
        assert escape_regex.fullmatch(spelling) is None, spelling
        assert invalid_escape_regex.fullmatch(spelling), spelling

    invalid_before_quote = invalid_escape_regex.match(r'\x0"')
    assert invalid_before_quote is not None
    assert invalid_before_quote.group() == r'\x0'

    malformed_before_escaped_quote = r'\x0\"'
    invalid_hex = invalid_escape_regex.match(malformed_before_escaped_quote)
    assert invalid_hex is not None
    assert invalid_hex.group() == r'\x0'
    assert escape_regex.fullmatch(malformed_before_escaped_quote[invalid_hex.end() :])

    numeric_pattern = numeric['match']
    assert isinstance(numeric_pattern, str)
    numeric_regex = re.compile(numeric_pattern)
    for spelling in (
        '0',
        '123',
        '-7',
        '+7',
        '0b111',
        '-0b111',
        '+0o222',
        '0x333',
        '-0x333',
        '+0xAbCd',
    ):
        assert numeric_regex.fullmatch(spelling), spelling
    for spelling in ('+', '-', '0b', '-0b', '0b2', '0o8', '0x', '+0x', '0xg'):
        assert numeric_regex.fullmatch(spelling) is None, spelling
    for source in ('0x1.2', '-0x1.2', '0b1-foo', 'name-123', 'name+123', '123abc'):
        assert numeric_regex.search(source) is None, source

    after_string = numeric_regex.search('"value"-0x2a ')
    assert after_string is not None
    assert after_string.group() == '-0x2a'


if __name__ == '__main__':
    main()
