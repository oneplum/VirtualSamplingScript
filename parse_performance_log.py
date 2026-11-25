#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, TextIO


@dataclass
class ParsedLabel:
    dataset_name: str
    image_kind: str
    level: int | None
    method: str
    lighting_enabled: bool
    virtual_sampling_method: str | None
    virtual_samples: int | None
    rotation: int
    baseline_source: str | None
    baseline_variant: str | None
    true_samples: int | None


@dataclass
class Measurement:
    label: str
    fps: int
    label_line_number: int
    fps_line_number: int
    vendor: str | None
    renderer: str | None
    parsed_label: ParsedLabel | None
    label_matches_expected_structure: bool
    label_parse_error: str | None


def warn(message: str, *, stream: TextIO) -> None:
    print(f"Warning: {message}", file=stream)


def _parse_int(token: str, *, field_name: str) -> tuple[int | None, str | None]:
    if token.isdigit():
        return int(token), None
    return None, f"expected integer {field_name}, got {token!r}"


def parse_measurement_label(label: str) -> tuple[ParsedLabel | None, str | None]:
    if not label.endswith(".png"):
        return None, "label does not end with .png"

    parts = label[:-4].split("-")
    if len(parts) < 4:
        return None, f"expected at least 4 name parts, got {len(parts)}"

    dataset_name = parts[0]
    if not dataset_name:
        return None, "dataset name is empty"

    if parts[1] == "baseline":
        if len(parts) == 4:
            method = parts[2]
            lighting_enabled = False
            rotation_token = parts[3]
        elif len(parts) == 5 and parts[3] == "l":
            method = parts[2]
            lighting_enabled = True
            rotation_token = parts[4]
        else:
            return None, (
                "baseline name does not match "
                "'volname-baseline-method[-l]-rotation.png'"
            )

        rotation, error = _parse_int(rotation_token, field_name="rotation")
        if error is not None:
            return None, error

        if method == "analytic":
            image_kind = "legacy_baseline_analytic"
            baseline_source = "analytic"
        else:
            image_kind = "legacy_baseline_discrete"
            baseline_source = "discrete"

        return ParsedLabel(
            dataset_name=dataset_name,
            image_kind=image_kind,
            level=None,
            method=method,
            lighting_enabled=lighting_enabled,
            virtual_sampling_method=None,
            virtual_samples=None,
            rotation=rotation,
            baseline_source=baseline_source,
            baseline_variant="legacy_baseline",
            true_samples=None,
        ), None

    level, error = _parse_int(parts[1], field_name="level")
    if error is not None:
        return None, error

    if len(parts) in (5, 6) and parts[2].startswith("baseline"):
        baseline_variant = parts[2]
        baseline_match = baseline_variant.removeprefix("baseline")
        true_samples, error = _parse_int(
            baseline_match, field_name="baseline true-sample count"
        )
        if error is not None:
            return None, error

        method = parts[3]
        if len(parts) == 5:
            lighting_enabled = False
            rotation_token = parts[4]
        elif parts[4] == "l":
            lighting_enabled = True
            rotation_token = parts[5]
        else:
            return None, (
                "baseline reference name does not match "
                "'volname-level-baselineN-method[-l]-rotation.png'"
            )

        rotation, error = _parse_int(rotation_token, field_name="rotation")
        if error is not None:
            return None, error

        baseline_source = "analytic" if method == "analytic" else "discrete"
        return ParsedLabel(
            dataset_name=dataset_name,
            image_kind="baseline_reference",
            level=level,
            method=method,
            lighting_enabled=lighting_enabled,
            virtual_sampling_method=None,
            virtual_samples=None,
            rotation=rotation,
            baseline_source=baseline_source,
            baseline_variant=baseline_variant,
            true_samples=true_samples,
        ), None

    if len(parts) == 4:
        method = parts[2]
        lighting_enabled = False
        rotation_token = parts[3]
        virtual_sampling_method = None
        virtual_samples = None
        image_kind = "downsampled"
    elif len(parts) == 5 and parts[3] == "l":
        method = parts[2]
        lighting_enabled = True
        rotation_token = parts[4]
        virtual_sampling_method = None
        virtual_samples = None
        image_kind = "downsampled"
    elif len(parts) == 6:
        method = parts[2]
        lighting_enabled = False
        virtual_sampling_method = parts[3]
        virtual_samples, error = _parse_int(parts[4], field_name="virtual sample count")
        if error is not None:
            return None, error
        rotation_token = parts[5]
        image_kind = "virtual_sampled"
    elif len(parts) == 7 and parts[3] == "l":
        method = parts[2]
        lighting_enabled = True
        virtual_sampling_method = parts[4]
        virtual_samples, error = _parse_int(parts[5], field_name="virtual sample count")
        if error is not None:
            return None, error
        rotation_token = parts[6]
        image_kind = "virtual_sampled"
    else:
        return None, (
            "non-baseline name does not match either "
            "'volname-level-method[-l]-rotation.png' or "
            "'volname-level-method[-l]-virtualMethod-subdiv-rotation.png'"
        )

    rotation, error = _parse_int(rotation_token, field_name="rotation")
    if error is not None:
        return None, error

    return ParsedLabel(
        dataset_name=dataset_name,
        image_kind=image_kind,
        level=level,
        method=method,
        lighting_enabled=lighting_enabled,
        virtual_sampling_method=virtual_sampling_method,
        virtual_samples=virtual_samples,
        rotation=rotation,
        baseline_source=None,
        baseline_variant=None,
        true_samples=None,
    ), None


def parse_performance_log(lines: Iterable[str], *, warning_stream: TextIO) -> list[Measurement]:
    measurements: list[Measurement] = []
    previous_line_text: str | None = None
    previous_line_number: int | None = None
    current_vendor: str | None = None
    current_renderer: str | None = None
    unmatched_label_occurrences: defaultdict[str, list[int]] = defaultdict(list)
    unmatched_label_reasons: dict[str, str] = {}

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.rstrip("\n")
        stripped_line = line.strip()

        if stripped_line.startswith("Vendor:"):
            vendor_value = stripped_line.split(":", 1)[1].strip()
            current_vendor = vendor_value or None
        elif stripped_line.startswith("Renderer:"):
            renderer_value = stripped_line.split(":", 1)[1].strip()
            current_renderer = renderer_value or None

        if line.endswith(" fps"):
            if (
                previous_line_text is None
                or previous_line_number is None
                or previous_line_text.strip() == ""
            ):
                warn(
                    f"missing previous line for fps entry at line {line_number}: {line!r}",
                    stream=warning_stream,
                )
            else:
                fps_text = line[: -len(" fps")].strip()
                if fps_text.isdigit():
                    parsed_label, label_parse_error = parse_measurement_label(previous_line_text)
                    if label_parse_error is not None:
                        unmatched_label_occurrences[previous_line_text].append(previous_line_number)
                        unmatched_label_reasons[previous_line_text] = label_parse_error

                    measurements.append(
                        Measurement(
                            label=previous_line_text,
                            fps=int(fps_text),
                            label_line_number=previous_line_number,
                            fps_line_number=line_number,
                            vendor=current_vendor,
                            renderer=current_renderer,
                            parsed_label=parsed_label,
                            label_matches_expected_structure=label_parse_error is None,
                            label_parse_error=label_parse_error,
                        )
                    )
                else:
                    warn(
                        f"invalid fps value at line {line_number}: {line!r}",
                        stream=warning_stream,
                    )
            # An fps-like line cannot serve as the descriptive label for the next entry.
            previous_line_text = None
            previous_line_number = None
        else:
            previous_line_text = line
            previous_line_number = line_number

    if unmatched_label_occurrences:
        total_occurrences = sum(len(line_numbers) for line_numbers in unmatched_label_occurrences.values())
        unique_labels = len(unmatched_label_occurrences)
        warn(
            (
                f"found {total_occurrences} measurement names that do not match the expected structure "
                f"across {unique_labels} unique labels"
            ),
            stream=warning_stream,
        )
        for index, label in enumerate(sorted(unmatched_label_occurrences), start=1):
            if index > 20:
                remaining = unique_labels - 20
                warn(
                    f"... plus {remaining} more unique unmatched labels",
                    stream=warning_stream,
                )
                break
            first_line_number = unmatched_label_occurrences[label][0]
            warn(
                (
                    f"unmatched measurement name at line {first_line_number}: {label!r} "
                    f"({unmatched_label_reasons[label]})"
                ),
                stream=warning_stream,
            )

    return measurements


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract performance measurements from a text log."
    )
    parser.add_argument("logfile", type=Path, help="Path to the performance log file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional JSON output path. Defaults to stdout.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="Indentation level for JSON output. Default: 2.",
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    with args.logfile.open("r", encoding="utf-8") as handle:
        measurements = parse_performance_log(handle, warning_stream=sys.stderr)

    payload = [asdict(measurement) for measurement in measurements]
    json_output = json.dumps(payload, indent=args.indent)

    if args.output is None:
        print(json_output)
    else:
        args.output.write_text(f"{json_output}\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
