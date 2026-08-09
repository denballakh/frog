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

    repository = dictionary(grammar['repository'])
    literals = dictionary(repository['literals'])
    string = named_pattern(patterns(literals['patterns']), 'string.quoted.double.frog')

    assert string['begin'] == '"'
    assert string['end'] == '"'
    assert 'match' not in string

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


if __name__ == '__main__':
    main()
