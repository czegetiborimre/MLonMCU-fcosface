"""
fix_bias88.py
=============
Post-synthesis patch for fcosface88.
Run from ai8x-synthesis/ after every ai8xize.py run.
Patches cnn_load_bias() in synthed_net/fcosface88/cnn.c to fix the
wide=True head layer bias scale (int8 shift=7 gives sigmoid=36%,
we overwrite with pre-shifted int32 values for correct sigmoid~27%).

Usage:
    python fix_bias88.py
    python fix_bias88.py --cnn-c path/to/cnn.c
"""

import re
import sys
import argparse

DEFAULT_CNN_C = "synthed_net/fcosface88/cnn.c"

PATCH_COMMENT = "/* fix_bias88.py patch */"

PATCH_LINES = """
  /* fix_bias88.py patch: head (layer 7) is wide=True so needs shift<<10,
   * not shift<<7. Overwrite the 5 head bias words written by memcpy_8to32
   * above with pre-shifted int32 values (original int8 * 1024). */
  *((volatile uint32_t *) 0x50908100) = 0xFFFEE400;  /* ch0 obj:  -71*1024=-72704 sigmoid~27% */
  *((volatile uint32_t *) 0x50908104) = 0xFFFFFC00;  /* ch1 reg:   -1*1024=  -1024 */
  *((volatile uint32_t *) 0x50908108) = 0x00000C00;  /* ch2 reg:    3*1024=   3072 */
  *((volatile uint32_t *) 0x5090810C) = 0x00000000;  /* ch3 reg:    0*1024=      0 */
  *((volatile uint32_t *) 0x50908110) = 0x00000800;  /* ch4 reg:    2*1024=   2048 */"""


def patch(cnn_c_path):
    with open(cnn_c_path, "r") as f:
        source = f.read()

    if PATCH_COMMENT in source:
        print(f"[fix_bias88] Already patched: {cnn_c_path}")
        return

    # Find the line containing bias_2 inside a memcpy call — flexible regex
    pattern = re.compile(r'(memcpy[^\n]*bias_2[^\n]*\n)', re.IGNORECASE)
    m = pattern.search(source)
    if not m:
        # Show what bias lines exist to help debug
        print(f"[fix_bias88] ERROR: could not find bias_2 memcpy line in {cnn_c_path}")
        print("[fix_bias88] Lines containing 'bias' in that file:")
        for i, line in enumerate(source.splitlines()):
            if 'bias' in line.lower() and 'memcpy' in line.lower():
                print(f"  line {i+1}: {line}")
        sys.exit(1)

    insert_pos = m.end()
    patched = (source[:insert_pos]
               + "  " + PATCH_COMMENT + "\n"
               + PATCH_LINES + "\n"
               + source[insert_pos:])

    with open(cnn_c_path, "w") as f:
        f.write(patched)

    print(f"[fix_bias88] Patched: {cnn_c_path}")
    print(f"[fix_bias88] Inserted head bias overwrite after bias_2 memcpy.")
    print(f"[fix_bias88] Copy cnn.c and weights.h to your STM32CubeIDE project and rebuild.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn-c", default=DEFAULT_CNN_C)
    args = parser.parse_args()
    patch(args.cnn_c)


if __name__ == "__main__":
    main()