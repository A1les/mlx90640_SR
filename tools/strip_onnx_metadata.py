"""Remove top-level ONNX node debug metadata without changing graph computation.

This small wire-format utility needs no ONNX dependency. It preserves every
field byte-for-byte except NodeProto.metadata_props (field 9), reached through
ModelProto.graph (7) and GraphProto.node (1). Field numbers follow the ONNX
specification: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
It is intended for this project's frozen, flat inference graph.
"""
import argparse
import hashlib
from pathlib import Path
import re


def varint(data, offset):
    value = 0
    for shift in range(0, 70, 7):
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if not byte & 128:
            return value, offset
    raise ValueError('Invalid protobuf varint')


def encode_varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def fields(data):
    offset = 0
    while offset < len(data):
        start = offset
        key, offset = varint(data, offset)
        number, wire = key >> 3, key & 7
        if wire == 2:
            size, offset = varint(data, offset)
            value = data[offset:offset + size]
            offset += size
        elif wire == 0:
            value, offset = varint(data, offset)
        elif wire in (1, 5):
            size = 8 if wire == 1 else 4
            value = data[offset:offset + size]
            offset += size
        else:
            raise ValueError(f'Unsupported protobuf wire type: {wire}')
        if offset > len(data) or number == 0:
            raise ValueError('Truncated or invalid protobuf field')
        yield number, wire, value, data[start:offset]


def strip_debug_metadata(data, level=0):
    output = bytearray()
    for number, wire, value, original in fields(data):
        if level == 2 and number == 9:
            continue
        descend = (level == 0 and number == 7) or (level == 1 and number == 1)
        if descend:
            if wire != 2:
                raise ValueError('Invalid ONNX graph field')
            value = strip_debug_metadata(value, level + 1)
            output += encode_varint(number * 8 + 2) + encode_varint(len(value)) + value
        else:
            output += original
    return bytes(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    original = args.source.read_bytes()
    cleaned = strip_debug_metadata(original)
    assert strip_debug_metadata(cleaned) == cleaned
    strings = re.findall(rb'[ -~]{12,}', cleaned)
    if any(re.search(rb'[A-Za-z]:[\\/]|/(?:home|Users)/', text) for text in strings):
        raise ValueError('Workstation paths remain outside the supported debug fields')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as stream:
        stream.write(cleaned)
    print(f'SHA256 {hashlib.sha256(cleaned).hexdigest()} ({len(cleaned)} bytes)')


if __name__ == '__main__':
    main()
