#!/usr/bin/env python3

import re
import sys
import glob
from pathlib import Path

HEADER = """\
// Auto-generated for AVR PROGMEM.
// Layouts follow Noodle: Conv [O][I][K][K], DW [C][K][K], FCN [O][I].

#pragma once
#include <avr/pgmspace.h>

"""

FOOTER = """\

// --------------------
"""

PATTERN = re.compile(
    r"\bstatic\s+const\s+float\s+([A-Za-z0-9_]+)\s*\[\]\s*=\s*\{",
    re.MULTILINE
)

INCLUDE_GUARD_RE = re.compile(
    r"^\s*#ifndef\s+\w+\s*\n\s*#define\s+\w+\s*\n",
    re.MULTILINE
)

ENDIF_RE = re.compile(
    r"\n\s*#endif\s*(?://.*)?\s*$",
    re.MULTILINE
)


def expand_inputs(args):
    files = []
    for arg in args:
        matches = glob.glob(arg)
        if matches:
            files.extend(matches)
        else:
            files.append(arg)

    paths = [Path(f) for f in files if Path(f).is_file()]
    return sorted(paths, key=lambda p: p.name)


def strip_header_wrapper(text):
    # Remove pragma once from input headers
    text = re.sub(r'^\s*#pragma\s+once\s*\n', '', text, flags=re.MULTILINE)

    # Remove include guards if present
    text = INCLUDE_GUARD_RE.sub("", text, count=1)
    text = ENDIF_RE.sub("\n", text, count=1)

    # Remove repeated pgmspace includes
    text = re.sub(r'^\s*#include\s+<avr/pgmspace\.h>\s*\n', '', text, flags=re.MULTILINE)

    return text


def convert_text(text):
    text = strip_header_wrapper(text)
    text = PATTERN.sub(r"const float \1[] PROGMEM = {", text)
    return text.strip() + "\n"


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python make_progmem.py output.h input1.h input2.h ...")
        print("  python make_progmem.py output.h '*.h'")
        sys.exit(1)

    output_file = Path(sys.argv[1]).resolve()
    input_files = expand_inputs(sys.argv[2:])

    input_files = [
        p for p in input_files
        if p.resolve() != output_file
    ]

    if not input_files:
        print("No input header files found.")
        sys.exit(1)

    pieces = [HEADER]

    for p in input_files:
        pieces.append("\n// ==================================================\n")
        pieces.append(f"// {p.name}\n")
        pieces.append("// ==================================================\n\n")
        pieces.append(convert_text(p.read_text()))

    pieces.append(FOOTER)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("".join(pieces))

    print(f"Written: {output_file}")
    print(f"Included {len(input_files)} file(s):")
    for p in input_files:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
