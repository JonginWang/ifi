from ifi.analysis.ifi_analyzer import normalize_cli_argv
from ifi.analysis.main_analysis import build_argument_parser


VEST_FIELDS = "1 13 18 21 25 4 5 6 59 60 61 62 63 64 65 101 102 109 112 113 138 139 140 141 142 143 144 171 192 214 216 217 218 236 257 258"


def _parse(argv):
    parser = build_argument_parser()
    return parser.parse_args(normalize_cli_argv(argv))


def test_ifi_analyzer_accepts_positional_query_with_quoted_numeric_arrays():
    args = _parse(
        [
            "47809:47840",
            "--freq",
            "94 280",
            "--density",
            "--vest-fields",
            VEST_FIELDS,
            "--plot",
            "--plot_envelope",
            "--no_plot_block",
            "--downsample",
            "200",
        ]
    )

    assert args.query == ["47809:47840"]
    assert args.freq == [94.0, 280.0]
    assert len(args.vest_fields) == 36
    assert args.vest_fields[:3] == [1, 13, 18]
    assert args.vest_fields[-3:] == [236, 257, 258]
    assert args.downsample == 200
    assert args.plot is True
    assert args.plot_envelope is True
    assert args.no_plot_block is True


def test_ifi_analyzer_accepts_query_option_with_unquoted_numeric_arrays():
    args = _parse(
        [
            "--query",
            "47809:47840",
            "--freq",
            "94",
            "280",
            "--density",
            "--vest-fields",
            "1",
            "13",
            "18",
            "--downsample",
            "200",
        ]
    )

    assert args.query == ["47809:47840"]
    assert args.freq == [94.0, 280.0]
    assert args.vest_fields == [1, 13, 18]
    assert args.downsample == 200


def test_ifi_analyzer_accepts_query_option_with_comma_numeric_arrays():
    args = _parse(
        [
            "--query",
            "47809:47840",
            "--freq",
            "94,280",
            "--vest-fields",
            "1,13,18",
            "--downsample",
            "200",
        ]
    )

    assert args.query == ["47809:47840"]
    assert args.freq == [94.0, 280.0]
    assert args.vest_fields == [1, 13, 18]
    assert args.downsample == 200


def test_ifi_analyzer_accepts_flip_density_aliases():
    hyphen_args = _parse(["47809", "--density", "--flip-density"])
    underscore_args = _parse(["47809", "--density", "--flip_density"])

    assert hyphen_args.flip_density is True
    assert underscore_args.flip_density is True
