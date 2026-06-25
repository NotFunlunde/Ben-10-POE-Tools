#!/usr/bin/env python3
"""
extract_pss.py - Extract Ben 10: Protector of Earth .pss (BIGB) archive files.

Supports Ben 10: Protector of Earth (PS2) and other games using the BIGB
container format (I-Ninja, Family Guy, Bionicle, Catwoman, etc.) by
High Voltage Software / Argonaut Games.

Usage:
    python extract_pss.py <input.pss> [output_dir]

The script extracts:
    - BIGB header metadata (name, author, build info)
    - Raw compressed blocks (block1 = RIFF WAVS metadata, block2 = asset data)
    - Uncompressed data block (block4)
    - PS2 VAG/ADPCM audio block (split into individual streams if possible)

Note: Blocks 1 and 2 use a proprietary compression algorithm that has not been
publicly reverse-engineered. They are extracted as raw compressed data. See the
README or inline comments for format details.

File format reference:
    - XentaxWiki: Catwoman PCM (covers BIGB variants)
    - ZenHAX: I-Ninja (PS2) GAME.WAD/.DIR thread
    - ZenHAX: Ben 10: Protector of Earth (PS2) PSS thread
"""

import argparse
import json
import os
import struct
import sys


BIGB_MAGIC = b"BIGB"
SECTOR_ALIGN = 0x800


def align_up(offset, alignment):
    """Round offset up to the next multiple of alignment."""
    return (offset + alignment - 1) & ~(alignment - 1)


def read_cstring(data, offset, max_len=256):
    """Read a null-terminated string from data at the given offset."""
    end = data.find(b"\x00", offset, offset + max_len)
    if end == -1:
        end = offset + max_len
    return data[offset:end].decode("ascii", errors="replace")


def validate_ps2_adpcm(data, sample_size=16, max_checks=100):
    """Check if data looks like PS2 VAG/ADPCM audio.

    PS2 ADPCM uses 16-byte blocks where the second byte (flags) should
    have a limited range of valid values (0-7 typically).
    """
    valid = 0
    total = 0
    for i in range(0, min(len(data), max_checks * sample_size), sample_size):
        if i + 1 >= len(data):
            break
        flag = data[i + 1]
        total += 1
        if flag <= 7:
            valid += 1
    if total == 0:
        return 0.0
    return valid / total


def scan_strings(data, min_length=4):
    """Scan binary data for readable ASCII strings."""
    strings = []
    current = bytearray()
    start = 0
    for i, b in enumerate(data):
        if 0x20 <= b < 0x7F:
            if not current:
                start = i
            current.append(b)
        else:
            if len(current) >= min_length:
                s = current.decode("ascii")
                if any(c.isalpha() for c in s):
                    strings.append((start, s))
            current = bytearray()
    if len(current) >= min_length:
        s = current.decode("ascii")
        if any(c.isalpha() for c in s):
            strings.append((start, s))
    return strings


def find_audio_streams(riff_data):
    """Parse RIFF WAVS metadata from block1 to find audio stream info.

    Block1 contains a compressed RIFF WAVS structure with entries for each
    sound in the archive. We scan for 'strm' and 'AMPC' chunk tags to
    extract stream offset/size information, and 'name' chunks for filenames.
    """
    streams = []
    names = []

    # Scan for name chunks (contain filenames)
    for ext in [b".wav\x00", b".amp\x00"]:
        pos = 0
        while True:
            idx = riff_data.find(ext, pos)
            if idx == -1:
                break
            start = idx
            while start > 0 and 0x20 <= riff_data[start - 1] < 0x7F:
                start -= 1
            name = riff_data[start : idx + len(ext) - 1].decode("ascii", errors="replace")
            names.append(name)
            pos = idx + 1

    return names


class BIGBHeader:
    """Parsed BIGB archive header."""

    def __init__(self):
        self.magic = b""
        self.header_size = 0
        self.entry_count = 0
        self.version = 0
        self.name = ""
        self.author = ""
        self.build_cmd = ""
        self.size_block3 = 0  # additional raw block (often 0)
        self.size_block4 = 0  # raw data block
        self.size_audio = 0  # audio block size
        self.decomp_size2 = 0  # decompressed size of block 2
        self.comp_size1 = 0  # compressed size of block 1
        self.comp_size2 = 0  # compressed size of block 2
        self.data_offset = 0  # where block data starts

    def __str__(self):
        return (
            f"BIGB Archive\n"
            f"  Name: {self.name}\n"
            f"  Author: {self.author}\n"
            f"  Entries: {self.entry_count}\n"
            f"  Version: {self.version}\n"
            f"  Build: {self.build_cmd}\n"
            f"  Block1 (compressed RIFF metadata): {self.comp_size1} bytes\n"
            f"  Block2 (compressed data): {self.comp_size2} bytes"
            f" -> {self.decomp_size2} bytes decompressed\n"
            f"  Block3 (raw): {self.size_block3} bytes\n"
            f"  Block4 (raw): {self.size_block4} bytes\n"
            f"  Audio block: {self.size_audio} bytes\n"
            f"  Data offset: 0x{self.data_offset:X}"
        )

    def to_dict(self):
        return {
            "name": self.name,
            "author": self.author,
            "entry_count": self.entry_count,
            "version": self.version,
            "build_command": self.build_cmd,
            "block1_compressed_size": self.comp_size1,
            "block2_compressed_size": self.comp_size2,
            "block2_decompressed_size": self.decomp_size2,
            "block3_size": self.size_block3,
            "block4_size": self.size_block4,
            "audio_size": self.size_audio,
            "data_offset": self.data_offset,
        }


def parse_header(f):
    """Parse the BIGB header from a file object.

    Header layout (little-endian):
        0x00: 4 bytes  - magic ("BIGB")
        0x04: uint32   - header_size (data_offset = header_size + 0x10)
        0x08: uint32   - entry_count
        0x0C: uint32   - version/ID
        0x10: 65 bytes - name (null-terminated, leading length byte)
        0x51: 43 bytes - author (null-terminated, leading length byte)
        0x78: uint32   - size_block3 (additional raw block, 0 if absent)
        0x7C: uint32   - size_block4 (raw data block)
        0x80: uint32   - size_audio (PS2 VAG/ADPCM audio)
        0x84: uint32   - decompressed size of block 2
        0x88: uint32   - compressed size of block 1
        0x8C: uint32   - compressed size of block 2
        0x94: 260 bytes - build command string (null-terminated)
    """
    hdr = BIGBHeader()

    f.seek(0)
    hdr.magic = f.read(4)
    if hdr.magic != BIGB_MAGIC:
        raise ValueError(f"Not a BIGB file: magic = {hdr.magic!r}")

    raw_header_size = struct.unpack("<I", f.read(4))[0]
    hdr.header_size = raw_header_size
    hdr.data_offset = raw_header_size + 0x10

    hdr.entry_count = struct.unpack("<I", f.read(4))[0]
    hdr.version = struct.unpack("<I", f.read(4))[0]

    # Name field: 65 bytes starting at 0x10
    f.seek(0x10)
    name_raw = f.read(65)
    # Skip leading length byte if present
    if name_raw[0] < 0x20 and name_raw[1] >= 0x20:
        hdr.name = read_cstring(name_raw, 1, 64)
    else:
        hdr.name = read_cstring(name_raw, 0, 65)

    # Author field: 43 bytes starting at 0x51
    f.seek(0x51)
    author_raw = f.read(43)
    if author_raw[0] < 0x20 and len(author_raw) > 1 and author_raw[1] >= 0x20:
        hdr.author = read_cstring(author_raw, 1, 42)
    else:
        hdr.author = read_cstring(author_raw, 0, 43)

    # Size fields at fixed offsets
    f.seek(0x78)
    hdr.size_block3 = struct.unpack("<I", f.read(4))[0]
    hdr.size_block4 = struct.unpack("<I", f.read(4))[0]
    hdr.size_audio = struct.unpack("<I", f.read(4))[0]
    hdr.decomp_size2 = struct.unpack("<I", f.read(4))[0]
    hdr.comp_size1 = struct.unpack("<I", f.read(4))[0]
    hdr.comp_size2 = struct.unpack("<I", f.read(4))[0]

    # Build command string at 0x94
    f.seek(0x94)
    build_raw = f.read(260)
    hdr.build_cmd = read_cstring(build_raw, 0, 260)

    return hdr


def extract(input_path, output_dir, verbose=True):
    """Extract all blocks from a BIGB (.pss) archive.

    Returns a dict with extraction results and metadata.
    """
    os.makedirs(output_dir, exist_ok=True)

    with open(input_path, "rb") as f:
        file_size = f.seek(0, 2)
        f.seek(0)

        # Parse header
        hdr = parse_header(f)
        if verbose:
            print(hdr)
            print(f"  File size: {file_size} bytes (0x{file_size:X})")
            print()

        results = {
            "header": hdr.to_dict(),
            "file_size": file_size,
            "extracted_blocks": [],
        }

        # Save header metadata
        meta_path = os.path.join(output_dir, "metadata.json")
        with open(meta_path, "w") as mf:
            json.dump(results["header"], mf, indent=2)
        if verbose:
            print(f"Saved metadata to {meta_path}")

        # Calculate block offsets
        offset = hdr.data_offset

        # Block 1: compressed RIFF WAVS metadata
        block1_offset = offset
        block1_size = hdr.comp_size1
        if block1_size > 0:
            f.seek(block1_offset)
            block1_data = f.read(block1_size)
            block1_path = os.path.join(output_dir, "block1_riff_metadata.bin")
            with open(block1_path, "wb") as bf:
                bf.write(block1_data)
            if verbose:
                print(
                    f"Extracted block1 (compressed RIFF metadata):"
                    f" {block1_size} bytes at 0x{block1_offset:X}"
                )

            # Try to extract readable info from block1
            names = find_audio_streams(block1_data)
            readable_strings = scan_strings(block1_data)

            if names:
                names_path = os.path.join(output_dir, "filenames.txt")
                with open(names_path, "w") as nf:
                    for name in names:
                        nf.write(name + "\n")
                if verbose:
                    print(f"  Found {len(names)} filename(s) in metadata")

            if readable_strings:
                strings_path = os.path.join(output_dir, "strings.txt")
                with open(strings_path, "w") as sf:
                    for spos, s in readable_strings:
                        sf.write(f"0x{spos:04X}: {s}\n")
                if verbose:
                    print(
                        f"  Extracted {len(readable_strings)} readable"
                        f" strings from block1"
                    )

            results["extracted_blocks"].append(
                {
                    "name": "block1_riff_metadata",
                    "offset": block1_offset,
                    "compressed_size": block1_size,
                    "type": "compressed",
                    "description": "RIFF WAVS metadata (proprietary compression)",
                }
            )

            offset += block1_size

        # Block 2: compressed asset data
        block2_offset = offset
        block2_size = hdr.comp_size2
        if block2_size > 0:
            f.seek(block2_offset)
            block2_data = f.read(block2_size)
            block2_path = os.path.join(output_dir, "block2_compressed_data.bin")
            with open(block2_path, "wb") as bf:
                bf.write(block2_data)
            if verbose:
                ratio = hdr.decomp_size2 / block2_size if block2_size > 0 else 0
                print(
                    f"Extracted block2 (compressed data):"
                    f" {block2_size} bytes at 0x{block2_offset:X}"
                    f" (decompresses to {hdr.decomp_size2} bytes,"
                    f" ratio {ratio:.2f}x)"
                )
            results["extracted_blocks"].append(
                {
                    "name": "block2_compressed_data",
                    "offset": block2_offset,
                    "compressed_size": block2_size,
                    "decompressed_size": hdr.decomp_size2,
                    "type": "compressed",
                    "description": "Asset data (proprietary compression)",
                }
            )
            offset += block2_size

        # Align to sector boundary
        offset = align_up(offset, SECTOR_ALIGN)

        # Block 3: additional raw data (often 0 / absent)
        if hdr.size_block3 > 0:
            block3_offset = offset
            f.seek(block3_offset)
            block3_data = f.read(hdr.size_block3)
            block3_path = os.path.join(output_dir, "block3_raw.bin")
            with open(block3_path, "wb") as bf:
                bf.write(block3_data)
            if verbose:
                print(
                    f"Extracted block3 (raw):"
                    f" {hdr.size_block3} bytes at 0x{block3_offset:X}"
                )
            results["extracted_blocks"].append(
                {
                    "name": "block3_raw",
                    "offset": block3_offset,
                    "size": hdr.size_block3,
                    "type": "raw",
                }
            )
            offset += hdr.size_block3
            offset = align_up(offset, SECTOR_ALIGN)

        # Block 4: raw data block
        if hdr.size_block4 > 0:
            block4_offset = offset
            f.seek(block4_offset)
            block4_data = f.read(hdr.size_block4)
            block4_path = os.path.join(output_dir, "block4_raw.bin")
            with open(block4_path, "wb") as bf:
                bf.write(block4_data)
            if verbose:
                print(
                    f"Extracted block4 (raw):"
                    f" {hdr.size_block4} bytes at 0x{block4_offset:X}"
                )
            results["extracted_blocks"].append(
                {
                    "name": "block4_raw",
                    "offset": block4_offset,
                    "size": hdr.size_block4,
                    "type": "raw",
                }
            )
            offset += hdr.size_block4
            offset = align_up(offset, SECTOR_ALIGN)

        # Audio block: PS2 VAG/ADPCM audio data (always at end of file)
        if hdr.size_audio > 0:
            audio_offset = file_size - hdr.size_audio

            if audio_offset >= offset:
                f.seek(audio_offset)
                audio_data = f.read(hdr.size_audio)

                # Validate ADPCM
                score = validate_ps2_adpcm(audio_data)

                audio_path = os.path.join(output_dir, "audio_ps2_adpcm.bin")
                with open(audio_path, "wb") as af:
                    af.write(audio_data)
                if verbose:
                    print(
                        f"Extracted audio block (PS2 VAG/ADPCM):"
                        f" {hdr.size_audio} bytes at 0x{audio_offset:X}"
                        f" (validation: {score:.1%})"
                    )

                results["extracted_blocks"].append(
                    {
                        "name": "audio_ps2_adpcm",
                        "offset": audio_offset,
                        "size": hdr.size_audio,
                        "type": "ps2_vag_adpcm",
                        "validation_score": score,
                    }
                )
            else:
                if verbose:
                    print(
                        f"Warning: audio block at 0x{audio_offset:X}"
                        f" overlaps data blocks (ends at 0x{offset:X})"
                    )

        # Check for gap between data blocks and audio
        audio_start = file_size - hdr.size_audio if hdr.size_audio > 0 else file_size
        if offset < audio_start:
            gap = audio_start - offset
            if verbose:
                print(
                    f"\nGap between data blocks and audio:"
                    f" {gap} bytes (0x{offset:X} - 0x{audio_start:X})"
                )

        # Save full extraction report
        report_path = os.path.join(output_dir, "extraction_report.json")
        with open(report_path, "w") as rf:
            json.dump(results, rf, indent=2)
        if verbose:
            print(f"\nSaved extraction report to {report_path}")

        return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract Ben 10: Protector of Earth .pss (BIGB) archive files.",
        epilog=(
            "This tool extracts raw blocks from BIGB archives. Compressed blocks"
            " (block1 and block2) are extracted as-is because the proprietary"
            " compression algorithm used by HVS/Argonaut has not been publicly"
            " reverse-engineered. The audio block (PS2 VAG/ADPCM) and block4"
            " (raw data) are extracted in their original uncompressed form."
        ),
    )
    parser.add_argument("input", help="Path to the .pss input file")
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help=(
            "Output directory (default: <input>_extracted)"
        ),
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress output messages"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output
    if output_dir is None:
        base = os.path.splitext(args.input)[0]
        output_dir = base + "_extracted"

    extract(args.input, output_dir, verbose=not args.quiet)


if __name__ == "__main__":
    main()
