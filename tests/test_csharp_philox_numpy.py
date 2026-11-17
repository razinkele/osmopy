"""NumPy reproduces the C# engine's Philox words (plan task 0.5).

The literals here are the same ones ``csharp/tests/Osmose.Engine.Tests/PhiloxTests.cs`` asserts
against the C# implementation, so the two languages are pinned to each other through them. The
first three are Random123's published known-answer vectors for philox4x64_10.
"""

from __future__ import annotations

import pytest

from osmose.engine_cs.philox import block, run_key, splitmix64, step_word

M = (1 << 64) - 1

RANDOM123 = [
    (
        (0, 0),
        (0, 0, 0, 0),
        [0x16554D9ECA36314C, 0xDB20FE9D672D0FDC, 0xD7E772CEE186176B, 0x7E68B68AEC7BA23B],
    ),
    (
        (M, M),
        (M, M, M, M),
        [0x87B092C3013FE90B, 0x438C3C67BE8D0224, 0x9CC7D7C69CD777B6, 0xA09CAEBF594F0BA0],
    ),
    (
        (0x452821E638D01377, 0xBE5466CF34E90C6C),
        (0x243F6A8885A308D3, 0x13198A2E03707344, 0xA4093822299F31D0, 0x082EFA98EC4E6C89),
        [0xA528F45403E61D95, 0x38C72DBD566E9788, 0xA5A1610E72FD18B5, 0x57BD43B5E52B7FE6],
    ),
]


@pytest.mark.parametrize(("key", "ctr", "expected"), RANDOM123)
def test_numpy_call_reproduces_random123_known_answers(key, ctr, expected):
    assert block(key, *ctr) == expected


def test_splitmix64_and_run_key():
    assert splitmix64(0) == 0xE220A8397B1DCDAF
    assert run_key(42) == (0xBDD732262FEB6E95, 0x4F534D4F53450001)
    assert run_key(-1)[0] == splitmix64(M)  # seeds are taken as unsigned 64-bit, as in C#


ENGINE_LAYOUT = [
    # block, entity, step, substep, purpose -> words, key run_key(42)
    (
        0,
        17,
        5,
        3,
        1,
        [0xBECD341680FB139C, 0xA6A81D01E0BBF1EC, 0x4FF9B5FD10E6288E, 0xBE64881C5C0AEA12],
    ),
    (
        1,
        17,
        5,
        3,
        1,
        [0x3CECBBFD3E374F5A, 0x1973133903269769, 0x6D4CA27A9A167D95, 0x59E3EF8DA1C25E93],
    ),
    (
        0,
        17,
        5,
        3,
        6,
        [0x9B42371466055372, 0xC42FAED7844D80DC, 0x3984EE50B3D807D0, 0x0C474D9EB9BA349C],
    ),
    (
        0,
        (1 << 40) + 3,
        1199,
        9,
        13,
        [0xC3FC78D7B29CCB47, 0x215FAC61F9197312, 0x0A4233F1C4C8B53F, 0x586EEC303EC9687D],
    ),
]


@pytest.mark.parametrize(("blk", "entity", "step", "substep", "purpose", "expected"), ENGINE_LAYOUT)
def test_engine_counter_layout(blk, entity, step, substep, purpose, expected):
    assert block(run_key(42), blk, entity, step_word(step, substep), purpose) == expected


def test_step_word_bounds():
    assert step_word(5, 3) == (5 << 16) | 3
    for bad in ((-1, 0), (0, 65536), (0, -1)):
        with pytest.raises(ValueError):
            step_word(*bad)
