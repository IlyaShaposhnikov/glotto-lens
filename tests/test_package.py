import re

import glotto_lens


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", glotto_lens.__version__)
