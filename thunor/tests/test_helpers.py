import pytest
from thunor.helpers import _strip_tags


@pytest.mark.parametrize(
    'text, expected',
    [
        ('plain text', 'plain text'),
        ('<b>bold</b>', 'bold'),
        ('<span style="color:red">text</span>', 'text'),
        ('before<br>after', 'beforeafter'),
        ('a &amp; b', 'a & b'),
        ('<b>one</b> and <i>two</i>', 'one and two'),
        ('', ''),
        (None, ''),
    ],
)
def test_strip_tags(text, expected):
    assert _strip_tags(text) == expected
